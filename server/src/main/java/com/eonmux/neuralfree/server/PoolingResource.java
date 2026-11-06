package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.ops.heads.LateInteraction;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The pooling endpoints. {@code POST /v1/embeddings} returns OpenAI's body, one pooled vector per input (text,
 * token ids or chat messages) from an embedding or pooling model, shaped by {@link EmbeddingOptions}; a model
 * without pooling refuses, as llama.cpp's does. Every request's inputs reach the engine as one batch (the batch
 * engine schedules their prompts into its steps beside any generation), so the routes answer the same bodies the
 * per-input forwards answer, only batched. On a CLIP dual encoder a conversation holding one
 * {@code image_url} part and no text embeds that image through the vision tower, in the space of the text
 * vectors (vLLM's {@code CLIPEmbeddingModel}); images carry no tokens, so usage counts none for them, and the
 * other models refuse image parts by name. {@code POST /embeddings} and
 * {@code /embedding} are llama.cpp's native form: an array of {@code {"index", "embedding"}} whose embedding is
 * {@code [vector]} for a pooled model and one unnormalized row per token without pooling, over llama.cpp's prompt
 * forms ({@code input} or {@code content}).
 * {@code POST /v1/rerank} (also {@code /rerank}, {@code /v1/reranking}, {@code /reranking}, {@code /v2/rerank})
 * scores documents against a query and orders them. A sequence classifier (RANK or a pooled classifier) scores
 * every pair (cross-encoder); a per-token embedding model the MaxSim of the two L2-normalized token-row sets
 * (late interaction); a pooled embedding model answers the cosine similarity of the pooled vectors
 * (bi-encoder), vLLM's scoring dispatch; a generative model refuses by name. The query and the documents take
 * vLLM's score forms: a string, or {@code {"content": [...]}} holding one image part or text parts.
 *
 * <p>The vLLM pooling tasks follow: {@code POST /score} (also {@code /v1/score}) scores two sides against each
 * other — per pair through the rank classifier (cross-encoder), as the late-interaction MaxSim of the per-token
 * rows, or as cosine similarities of pooled embeddings (bi-encoder), the sides in the same string or content
 * forms; {@code POST /classify} answers a sequence classifier's scores or probabilities per input; {@code POST
 * /pooling} runs one pooling task ({@code embed}, {@code token_embed}, {@code token_classify} or {@code
 * classify}) over the completion or chat input forms — on a SPLADE model {@code embed} answers the sparse
 * embedding and {@code token_embed} the plain normalized rows, and a token_classify request takes vLLM's
 * {@code step_tag_id} and {@code returned_token_ids}
 * process-reward filters; {@code POST
 * /generative_scoring} reads a generative model's next-token probabilities of {@code label_token_ids} after
 * query and item (vLLM's generative scoring).</p>
 *
 * <p>Every pooling request selects LoRA adapters as a generation request does ({@link LoraAdmin#selection}):
 * llama.cpp's {@code lora} list, vLLM's {@code model} naming an adapter (then the response's {@code model}), or the
 * global scales; its forwards run in that adapter slot. An activated LoRA refuses, since a pooling forward runs
 * every row in one slot, and so does the CLIP dual encoder, which runs no adapters.</p>
 *
 * <p>The requests take vLLM's pooling fields explicitly: a score or rerank pair applies the task {@code
 * instruction} through the model's {@code {instruction}} score template (refused when the model has no place for
 * it) and {@code chat_template_kwargs} beside it, truncates per side ({@code max_tokens_per_query}, {@code
 * max_tokens_per_doc}) and the composed pair ({@code truncate_prompt_tokens}, {@code truncation_side}), and
 * applies or skips the classifier's final probability activation per request ({@code use_activation}: the
 * cross-encoder and classification head's activation, false for the raw head output; a bi-encoder's cosine reads
 * none); every endpoint validates {@code cache_salt} and {@code priority}, echoes the {@code X-Request-Id}
 * header, else the body {@code request_id}, as the response id and refuses {@code padding} beyond {@code
 * do_not_pad} and a non-empty {@code mm_processor_kwargs} by name, whose engine effects this server does not run;
 * the removed {@code normalize} and pooling tasks refuse where vLLM refuses them.</p>
 */
@Path("/")
public class PoolingResource {

    @Inject
    EngineService service;

    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public PoolingResource(EngineService service) {
        this(service, (ModelRouter) null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public PoolingResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated request keeps its client link. */
    public PoolingResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public PoolingResource() {
    }

    /** A routed copy: the target model's service, serving the same HTTP request (its {@code X-Request-Id}). */
    private PoolingResource(EngineService service, HttpServerRequest http) {
        this.service = service;
        this.http = http;
    }

    /** The resource over the service the body's {@code model} names, answering this request. */
    private PoolingResource routed(JsonView body) {
        return new PoolingResource(ModelRouter.target(router, service, body, null), http);
    }

    @POST
    @Path("/v1/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response embeddings(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.embeddingsOf(body));
    }

    /** llama.cpp's native embeddings. */
    @POST
    @Path("/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response nativeEmbeddings(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.nativeEmbeddingsOf(body));
    }

    /** llama.cpp's legacy spelling of the native embeddings. */
    @POST
    @Path("/embedding")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response nativeEmbedding(String bodyText) {
        return nativeEmbeddings(bodyText);
    }

    @POST
    @Path("/v1/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerank(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.rerankOf(body));
    }

    @POST
    @Path("/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankAlias(String bodyText) {
        return rerank(bodyText);
    }

    @POST
    @Path("/v1/reranking")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankingAlias(String bodyText) {
        return rerank(bodyText);
    }

    @POST
    @Path("/reranking")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankingShortAlias(String bodyText) {
        return rerank(bodyText);
    }

    /** vLLM's second versioned rerank alias. */
    @POST
    @Path("/v2/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankV2Alias(String bodyText) {
        return rerank(bodyText);
    }

    /** vLLM's score endpoint; {@code /v1/score} is its versioned alias. */
    @POST
    @Path("/score")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response score(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.scoreOf(body));
    }

    @POST
    @Path("/v1/score")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response scoreV1(String bodyText) {
        return score(bodyText);
    }

    /** vLLM's classification endpoint. */
    @POST
    @Path("/classify")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response classify(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.classifyOf(body));
    }

    /** vLLM's generic pooling endpoint. */
    @POST
    @Path("/pooling")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response pooling(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.poolingOf(body));
    }

    /** vLLM's generative scoring endpoint. */
    @POST
    @Path("/generative_scoring")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response generativeScoring(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        PoolingResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.generativeScoringOf(body));
    }

    private Response embeddingsOf(JsonView body) {
        String requestId = poolingRequestFields(body);
        removedPoolingTask(body);
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        if (service.perTokenEmbeddings() && service.sparsePooling() == null
                && service.poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            throw ServerJson.badRequest("Pooling type 'none' is not OAI compatible. Please use a different pooling"
                    + " type, or the native /embeddings endpoint");
        }
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength(),
                service.sparsePooling() != null);
        List<PoolingInputs.Input> inputs = PoolingInputs.of(body, service, options);
        Embedded[] embedded = embeddedAll(lora, inputs);
        List<Map<String, Object>> data = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("object", "embedding");
            entry.put("index", i);
            entry.put("embedding", options.encoded(options.shaped(embedded[i].vector())));
            data.add(entry);
            tokens += embedded[i].tokens();
        }
        return listEnvelope("embd-", requestId, model, data, tokens);
    }

    /** One pooled input: its vector, and the prompt rows it counts (0 for an image, its rows for a media prompt). */
    record Embedded(float[] vector, long tokens) {
    }

    /**
     * The pooled embedding of every input, in input order: the text inputs go to the service as one batch (the
     * batch engine runs their prompts in its steps), an image embeds through the dual encoder's vision tower and
     * a media conversation as one media prompt through the projector, each on its own.
     */
    Embedded[] embeddedAll(LoraSelection lora, List<PoolingInputs.Input> inputs) {
        Embedded[] out = new Embedded[inputs.size()];
        List<Integer> textAt = new ArrayList<>();
        List<int[]> batch = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            PoolingInputs.Input input = inputs.get(i);
            if (input.isImage()) {
                out[i] = new Embedded(image(lora, input.image(), input.field()), 0);
            } else if (input.isMedia()) {
                out[i] = embeddedOf(lora, input);
            } else {
                textAt.add(i);
                batch.add(input.tokens());
            }
        }
        float[][] pooled;
        try {
            pooled = service.embedAll(lora, batch.toArray(new int[0][]));
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
        for (int b = 0; b < pooled.length; b++) {
            out[textAt.get(b)] = new Embedded(pooled[b], batch.get(b).length);
        }
        return out;
    }

    /**
     * The pooled vector of one input under {@code lora}: an image through the dual encoder's vision tower, a media
     * conversation as one media prompt through the projector, text through the model.
     */
    Embedded embeddedOf(LoraSelection lora, PoolingInputs.Input input) {
        if (input.isImage()) {
            return new Embedded(image(lora, input.image(), input.field()), 0);
        }
        if (input.isMedia()) {
            MediaPrompt prompt = pooledMediaPrompt(input.chat(), ChatPlan.mediaLoras(service, lora));
            try {
                return new Embedded(service.embedMedia(lora, prompt), prompt.nTokens());
            } catch (UnsupportedOperationException e) {
                throw ServerJson.badRequest(e.getMessage());
            } catch (IOException e) {
                throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
            }
        }
        return new Embedded(pooled(lora, input.tokens()), input.tokens().length);
    }

    /** The dual encoder's embedding of one image under {@code lora}, a refusal naming it mapped to a 400. */
    private float[] image(LoraSelection lora, String reference, String field) {
        try {
            return service.embedImage(lora, reference, field);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /**
     * The {@link MediaPrompt} of one media conversation: the loaded image and audio parts, the template rendered
     * with media markers, and the encoded text segments with the parts' rows spliced between them — under the
     * request's adapters {@code mediaLoras}, whose identity the parts' cache entries and prompt chunks carry.
     */
    private MediaPrompt pooledMediaPrompt(ChatRequest chat, Supplier<LoraSlotKey> mediaLoras) {
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, chat, mediaLoras);
        String rendered = service.formatChat(chat, true);
        return ChatPlan.mediaPrompt(service, chat, rendered, items, mediaLoras);
    }

    /** llama.cpp's native body: per input {@code index} and {@code embedding} as a list of vectors. */
    private Response nativeEmbeddingsOf(JsonView body) {
        Object raw = body.has("input") ? body.raw("input") : body.raw("content");
        if (raw == null) {
            throw ServerJson.badRequest("\"input\" or \"content\" must be provided");
        }
        LoraSelection lora = service.loras().selection(body, service.loras().model(body));
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength(),
                service.sparsePooling() != null);
        boolean perToken = service.perTokenEmbeddings() && service.sparsePooling() == null
                && service.poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE;
        List<int[]> prompts = LlamaPrompts.of(raw, service);
        int[][] truncated = new int[prompts.size()][];
        for (int i = 0; i < prompts.size(); i++) {
            truncated[i] = options.truncated(prompts.get(i));
        }
        float[][] pooled = null;
        float[][][] rows = null;
        try {
            if (perToken) {
                rows = service.embedTokensAll(lora, truncated);
            } else {
                pooled = service.embedAll(lora, truncated);
            }
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            List<Object> vectors = new ArrayList<>();
            if (perToken) {
                for (float[] row : rows[i]) {
                    vectors.add(EmbeddingOptions.values(row));
                }
            } else {
                vectors.add(EmbeddingOptions.values(options.shaped(pooled[i])));
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("embedding", vectors);
            out.add(entry);
        }
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * vLLM's score forms: two sides named {@code queries} + {@code documents}, {@code queries} + {@code items},
     * {@code data_1} + {@code data_2} or {@code text_1} + {@code text_2}, each a {@link ScoreSides} side. A RANK
     * classifier scores every text pair (cross-encoder); a per-token embedding model the MaxSim of the two
     * L2-normalized token-row sets (late interaction); an embedding model answers the cosine similarity of the
     * pooled vectors (bi-encoder), an image entry embedded by a CLIP dual encoder's vision tower, and a SigLIP dual
     * encoder its sigmoid score of that cosine ({@link EngineService#biEncoderScore}). One query is scored against
     * every document; otherwise the sides must have the same length.
     */
    private Response scoreOf(JsonView body) {
        String requestId = poolingRequestFields(body);
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        boolean activation = OpenAiRequests.useActivation(body);
        ScoringOptions options = ScoringOptions.of(body, service);
        String[] pair = ScoreSides.scorePair(body);
        List<ScoreSides.Entry> one = ScoreSides.side(body.raw(pair[0]), pair[0]);
        List<ScoreSides.Entry> two = ScoreSides.side(body.raw(pair[1]), pair[1]);
        refuseMediaTruncation(one, two, options);
        if (one.size() != 1 && one.size() != two.size()) {
            throw ServerJson.badRequest(pair[0] + " and " + pair[1] + " must have the same length, or "
                    + pair[0] + " must hold one entry, got " + one.size() + " and " + two.size());
        }
        ScoringMode mode = scoringMode("scoring");
        String instruction = instructionOf(options, mode);
        int n = one.size() == 1 ? two.size() : one.size();
        double[] scores = new double[n];
        long tokens = 0;
        if (mode == ScoringMode.LATE) {
            float[][][] queries = sideTokenRows(lora, one.size() == 1 ? List.of(one.get(0)) : one,
                    options.maxQuery());
            float[][][] documents = sideTokenRows(lora, two, options.maxDoc());
            for (int i = 0; i < n; i++) {
                scores[i] = LateInteraction.maxSim(one.size() == 1 ? queries[0] : queries[i], documents[i]);
                tokens += sideTokens(one.get(one.size() == 1 ? 0 : i), options.maxQuery())
                        + sideTokens(two.get(i), options.maxDoc());
            }
        } else if (mode == ScoringMode.CROSS) {
            // the text pairs score as one batched rerank request; a pair with media parts scores alone through
            // the projector (the batched request takes token inputs only)
            List<int[]> batch = new ArrayList<>();
            List<Integer> textAt = new ArrayList<>();
            List<Integer> mediaAt = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                ScoreSides.Entry query = one.get(one.size() == 1 ? 0 : i);
                ScoreSides.Entry document = two.get(i);
                if (query.isMedia() || document.isMedia()) {
                    mediaAt.add(i);
                } else {
                    textAt.add(i);
                    batch.add(crossInput(query, document, options));
                }
            }
            float[] answers = crossScores(lora, batch, activation);
            for (int b = 0; b < answers.length; b++) {
                scores[textAt.get(b)] = answers[b];
                tokens += batch.get(b).length;
            }
            for (int i : mediaAt) {
                PairScore media = mediaCrossScore(lora, one.get(one.size() == 1 ? 0 : i), two.get(i), options,
                        activation);
                scores[i] = media.score();
                tokens += media.tokens();
            }
        } else {
            float[][] queries = sideVectors(lora, one.size() == 1 ? List.of(one.get(0)) : one, options.maxQuery());
            float[][] documents = sideVectors(lora, two, options.maxDoc());
            for (int i = 0; i < n; i++) {
                float[] query = one.size() == 1 ? queries[0] : queries[i];
                scores[i] = service.biEncoderScore(cosine(query, documents[i]));
                tokens += sideTokens(one.get(one.size() == 1 ? 0 : i), options.maxQuery())
                        + sideTokens(two.get(i), options.maxDoc());
            }
        }
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("object", "score");
            entry.put("score", scores[i]);
            data.add(entry);
        }
        return listEnvelope("score-", requestId, model, data, tokens);
    }

    /** The tokens of one text entry after {@code max} truncation (0 keeps all). */
    private int[] scoreTokens(String text, int max) {
        int[] ids = service.encode(text, false, false);
        return max > 0 && ids.length > max ? java.util.Arrays.copyOf(ids, max) : ids;
    }

    /** The prompt tokens one entry counts: its truncated text tokens, none for an image. */
    private long sideTokens(ScoreSides.Entry entry, int max) {
        return entry.isImage() ? 0 : scoreTokens(entry.text(), max).length;
    }

    /**
     * The embedding of every entry under {@code lora}, in entry order: the text entries as one batch (the batch
     * engine runs their prompts in its steps), an image through the dual encoder's vision tower and a media
     * conversation through the projector, each on its own. A text entry truncates to {@code max} content tokens
     * first, as vLLM truncates a score side before pooling it.
     */
    private float[][] sideVectors(LoraSelection lora, List<ScoreSides.Entry> entries, int max) {
        float[][] vectors = new float[entries.size()][];
        List<Integer> textAt = new ArrayList<>();
        List<int[]> batch = new ArrayList<>();
        for (int i = 0; i < entries.size(); i++) {
            ScoreSides.Entry entry = entries.get(i);
            if (entry.isImage()) {
                vectors[i] = image(lora, entry.image(), entry.field());
                continue;
            }
            if (entry.isMedia() && !service.dualEncoder()) {
                vectors[i] = mediaVector(lora, sideChat(entry));
                continue;
            }
            if (entry.isMedia()) {
                ScoreSides.requireClipForm(entry); // refuses unless the entry is text-only, which pools as text
            }
            int[] ids = service.encode(entry.text(), true, true);
            if (ids.length == 0) {
                throw ServerJson.badRequest(entry.field() + ": Input content cannot be empty");
            }
            textAt.add(i);
            batch.add(cappedTextTokens(entry.text(), max));
        }
        float[][] pooled;
        try {
            pooled = service.embedAll(lora, batch.toArray(new int[0][]));
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
        for (int b = 0; b < pooled.length; b++) {
            vectors[textAt.get(b)] = pooled[b];
        }
        return vectors;
    }

    /**
     * The prompt tokens of one text: its full tokens with the vocabulary's specials, or, when {@code max} cuts the
     * content, the first {@code max} content tokens decoded and re-encoded with them.
     */
    private int[] cappedTextTokens(String text, int max) {
        int[] ids = service.encode(text, true, true);
        if (ids.length == 0) {
            throw ServerJson.badRequest("Input content cannot be empty");
        }
        if (max <= 0) {
            return ids;
        }
        int[] content = service.encode(text, false, false);
        return content.length <= max ? ids
                : service.encode(service.tokenizer().decode(java.util.Arrays.copyOf(content, max)), true, true);
    }

    /** The text of at most {@code max} content tokens: the first tokens kept, decoded when the cap cuts. */
    private String truncatedText(String text, int max) {
        if (max <= 0) {
            return text;
        }
        int[] content = service.encode(text, false, false);
        return content.length <= max ? text : service.tokenizer().decode(java.util.Arrays.copyOf(content, max));
    }

    /**
     * The per-token rows of every text entry under {@code lora}, one batched request for the late-interaction
     * scorer: an image or media entry refuses by name (the scorer reads text only) and a text entry truncates to
     * {@code max} content tokens first, as vLLM truncates a score side.
     */
    private float[][][] sideTokenRows(LoraSelection lora, List<ScoreSides.Entry> entries, int max) {
        List<int[]> batch = new ArrayList<>();
        for (ScoreSides.Entry entry : entries) {
            if (entry.isImage() || entry.isMedia()) {
                throw ServerJson.badRequest(entry.field() + " holds an image or audio part; the late-interaction"
                        + " model " + service.architecture() + " reads text only");
            }
            batch.add(cappedTextTokens(entry.text(), max));
        }
        try {
            return service.embedTokensAll(lora, batch.toArray(new int[0][]));
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** The pooled vector of one media conversation under {@code lora} ({@link #embeddedOf} over its prompt). */
    private float[] mediaVector(LoraSelection lora, ChatRequest chat) {
        return embeddedOf(lora, new PoolingInputs.Input(null, null, chat, "input")).vector();
    }

    /** A one-message user conversation whose content is the raw parts of {@code entry}. */
    private ChatRequest sideChat(ScoreSides.Entry entry) {
        Map<String, Object> chat = new LinkedHashMap<>();
        List<Object> messages = new ArrayList<>();
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", entry.content());
        messages.add(message);
        chat.put("messages", messages);
        return OpenAiRequests.chat(JsonView.of(chat, entry.field()), service.options());
    }

    /** The scorer of a score/rerank request, over the model the service holds. */
    private enum ScoringMode {
        /** A RANK or other sequence classifier scores every composed pair (cross-encoder). */
        CROSS,
        /** The sides' L2-normalized per-token rows score by MaxSim (late interaction). */
        LATE,
        /** The cosine similarity of the two pooled vectors (bi-encoder). */
        BI
    }

    /**
     * The scorer of {@code endpoint}: a RANK classifier without a decision head answers the pairs (cross-encoder), a
     * per-token embedding model scores its token-row sets by late interaction, a pooled embedding model the cosine
     * (bi-encoder); the other models refuse by name.
     */
    private ScoringMode scoringMode(String endpoint) {
        if (service.decisionHead()) {
            throw ServerJson.badRequest(service.architecture() + " is a decision model; " + endpoint
                    + " needs a sequence classifier or an embedding model");
        }
        if (service.perTokenEmbeddings() && service.sparsePooling() == null
                && service.poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            return ScoringMode.LATE;
        }
        if (service.sequenceClassifier()) {
            return ScoringMode.CROSS;
        }
        if (service.poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            throw ServerJson.badRequest(endpoint + " needs pooled embeddings, a sequence classifier or per-token"
                    + " embeddings; " + service.architecture() + " (" + service.supportClass() + ", pooling "
                    + service.poolingKind() + ") stores none of them");
        }
        return ScoringMode.BI;
    }

    /**
     * The request's task instruction, refused when no score prompt of this model can place it: a bi-encoder and a
     * late-interaction model read none, and a cross-encoder needs the {@code tokenizer.chat_template.rerank} prompt
     * to hold an {@code {instruction}} placeholder beside {@code query} and {@code document}.
     */
    private String instructionOf(ScoringOptions options, ScoringMode mode) {
        String instruction = options.instruction();
        if (instruction == null) {
            return null;
        }
        if (mode == ScoringMode.LATE) {
            throw ServerJson.badRequest("instruction needs a cross-encoder; the late-interaction model scores by"
                    + " MaxSim and reads no instruction");
        }
        if (mode != ScoringMode.CROSS) {
            throw ServerJson.badRequest("instruction needs a cross-encoder; the embedding model scores by cosine"
                    + " and reads no instruction");
        }
        String template = service.tokenizer().rerankTemplate();
        if (template == null) {
            throw ServerJson.badRequest("instruction needs the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE + " prompt to place it; "
                    + service.architecture() + " has none");
        }
        if (template.indexOf(com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.RERANK_INSTRUCTION) < 0) {
            throw ServerJson.badRequest("instruction has no place in the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE
                    + " prompt; it applies through an {instruction} placeholder");
        }
        return instruction;
    }

    /** One cross-encoder score and the tokens of its pair input. */
    private record PairScore(double score, long tokens) {
    }

    /**
     * The cross-encoder input of one text pair: the sides truncate per their caps and compose through the model's
     * rerank prompt with the instruction ({@link #pairInput}), then the pair truncation cuts; an image entry
     * refuses by name (a media pair scores alone through the projector, as {@link #mediaCrossScore}).
     */
    private int[] crossInput(ScoreSides.Entry query, ScoreSides.Entry document, ScoringOptions options) {
        for (ScoreSides.Entry entry : new ScoreSides.Entry[]{query, document}) {
            if (entry.isImage()) {
                throw ServerJson.badRequest(entry.field() + " holds an image, but the RANK classifier "
                        + service.architecture() + " reads text only; image scoring needs a CLIP dual encoder");
            }
        }
        int[] input = pairInput(query.text(), document.text(), options);
        if (input.length == 0) {
            throw ServerJson.badRequest("Input content cannot be empty");
        }
        return input;
    }

    /**
     * The cross-encoder score of every text pair input, one batched request over the RANK classifier, the head's
     * final probability activation applied or skipped per {@code activation} (vLLM's {@code use_activation}).
     */
    private float[] crossScores(LoraSelection lora, List<int[]> inputs, boolean activation) {
        try {
            return service.rerankScores(lora, inputs.toArray(new int[0][]), activation);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the rerank forward failed: " + e.getMessage(), "server_error");
        }
    }

    /**
     * The pair prompt of one text pair: with the model's rerank prompt the side texts truncate first (vLLM cuts
     * each side before composing) and the prompt fills with query, document and instruction; without one the
     * truncated token arrays pack with the vocabulary's specials. {@code truncate_prompt_tokens} then cuts the
     * composed pair.
     */
    private int[] pairInput(String query, String document, ScoringOptions options) {
        int[] input;
        if (service.tokenizer().rerankTemplate() != null) {
            input = service.rerankInput(truncatedText(query, options.maxQuery()),
                    truncatedText(document, options.maxDoc()), options.instruction());
        } else {
            input = service.rerankInput(scoreTokens(query, options.maxQuery()),
                    scoreTokens(document, options.maxDoc()));
        }
        return options.truncation().truncated(input);
    }

    /** The cross-encoder score of one media pair, and the rows its prompt counted. */
    private PairScore mediaCrossScore(LoraSelection lora, ScoreSides.Entry query, ScoreSides.Entry document,
                                      ScoringOptions options, boolean activation) {
        if (options.maxQuery() > 0 || options.maxDoc() > 0) {
            throw ServerJson.badRequest("max_tokens_per_query and max_tokens_per_doc truncate text entries only;"
                    + " a media entry keeps its rows");
        }
        if (!service.media().enabled()) {
            ScoreSides.Entry media = query.isMedia() ? query : document;
            throw ServerJson.badRequest(media.field() + " holds an image or audio part, but the RANK classifier "
                    + service.architecture() + " reads text only; media scoring needs a media projector ("
                    + MediaOptions.MMPROJ_PROPERTY + ")");
        }
        MediaPrompt pair = rerankMediaPrompt(query, document, options.instruction(),
                ChatPlan.mediaLoras(service, lora));
        try {
            return new PairScore(service.rerankScoreMedia(lora, pair, activation), pair.nTokens());
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the rerank forward failed: " + e.getMessage(), "server_error");
        }
    }

    /**
     * The media prompt of one query-document pair: the model's rerank prompt fills with the sides' text (each
     * side's media markers ahead of its text, {@link #inlineText}), the parts load in the prompt's marker order —
     * the side the template places first contributes its parts first, as jina_vl's processor reverses vLLM's
     * query-first items for its document-first template — and the model's rerank suffix ids close the prompt.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when the model carries no rerank prompt, or the rendered
     *         prompt does not pair its markers with the parts
     */
    private MediaPrompt rerankMediaPrompt(ScoreSides.Entry query, ScoreSides.Entry document, String instruction,
                                          Supplier<LoraSlotKey> mediaLoras) {
        String template = service.tokenizer().rerankTemplate();
        if (template == null) {
            throw ServerJson.badRequest("media scoring needs the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE
                    + " prompt to place the query and document markers; " + service.architecture() + " has none");
        }
        if (instruction != null) {
            template = template.replace(com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.RERANK_INSTRUCTION,
                    instruction);
        }
        boolean documentFirst = template.indexOf("{document}") < template.indexOf("{query}");
        String rendered = template.replace("{query}", inlineText(query))
                .replace("{document}", inlineText(document));
        ScoreSides.Entry first = documentFirst ? document : query;
        ScoreSides.Entry second = documentFirst ? query : document;
        List<Object> messages = new ArrayList<>();
        for (ScoreSides.Entry entry : new ScoreSides.Entry[]{first, second}) {
            if (entry.content() != null) {
                Map<String, Object> message = new LinkedHashMap<>();
                message.put("role", "user");
                message.put("content", entry.content());
                messages.add(message);
            }
        }
        Map<String, Object> chat = new LinkedHashMap<>();
        chat.put("messages", messages);
        ChatRequest request = OpenAiRequests.chat(JsonView.of(chat, "score sides"), service.options());
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, request, mediaLoras);
        MediaPrompt prompt;
        try {
            prompt = service.media().prompt(rendered, items, List.of(), -1, text -> service.encode(text, false, true),
                    mediaLoras);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        int[] suffix = service.tokenizer().rerankSuffix();
        if (suffix.length == 0) {
            return prompt;
        }
        MediaPrompt.Builder withSuffix = MediaPrompt.builder();
        prompt.chunks().forEach(withSuffix::add);
        return withSuffix.appendText(suffix).build();
    }

    /**
     * One side's text as the pair prompt reads it, as vLLM's score content parser builds it: the string itself, or a
     * media marker per image or audio part (placed first, with no separator) followed by the text parts joined with
     * newlines.
     */
    private static String inlineText(ScoreSides.Entry entry) {
        if (entry.content() == null) {
            return entry.text();
        }
        StringBuilder out = new StringBuilder();
        for (Object raw : entry.content()) {
            if (raw instanceof Map<?, ?> part && isMediaPart(part.get("type"))) {
                out.append(ChatMessages.MEDIA_MARKER);
            }
        }
        return out.append(entry.text()).toString();
    }

    private static boolean isMediaPart(Object type) {
        return ChatMessages.PART_IMAGE.equals(type) || ChatMessages.PART_AUDIO.equals(type)
                || ChatMessages.PART_AUDIO_URL.equals(type);
    }

    /** The cosine similarity of two vectors, 0 when either is zero. */
    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            dot += (double) a[i] * b[i];
            normA += (double) a[i] * a[i];
            normB += (double) b[i] * b[i];
        }
        double norm = Math.sqrt(normA) * Math.sqrt(normB);
        return norm == 0 ? 0 : dot / norm;
    }

    /** vLLM's classification endpoint: one sequence classifier output per input, as probabilities. */
    private Response classifyOf(JsonView body) {
        String requestId = poolingRequestFields(body);
        removedPoolingTask(body);
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        boolean activation = OpenAiRequests.useActivation(body);
        requireSequenceClassifier();
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength());
        List<int[]> inputs = PoolingInputs.tokens(body, service, options, "classification");
        ClassifierOutput[] outputs = classifyBatch(lora, inputs.toArray(new int[0][]), activation);
        List<Map<String, Object>> data = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int[] ids = inputs.get(i);
            tokens += ids.length;
            ClassifierOutput output = outputs[i];
            float[] probs = activation ? classScores(output) : output.scores();
            int best = 0;
            for (int j = 1; j < probs.length; j++) {
                if (probs[j] > probs[best]) {
                    best = j;
                }
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("label", output.label(best));
            entry.put("probs", EmbeddingOptions.values(probs));
            entry.put("num_classes", probs.length);
            data.add(entry);
        }
        return listEnvelope("classify-", requestId, model, data, tokens);
    }

    /**
     * The scores {@code /classify} reports: a one-output sequence classifier's raw score (a softmax of one output
     * would flatten it to 1), otherwise the probabilities.
     */
    private static float[] classScores(ClassifierOutput output) {
        return output.size() == 1 ? output.scores() : output.probabilities();
    }

    /** The sequence classifier output of every input, one batched request, the head's activation applied. */
    private ClassifierOutput[] classifyBatch(LoraSelection lora, int[][] inputs) {
        return classifyBatch(lora, inputs, true);
    }

    /**
     * {@link #classifyBatch} with the head's final probability activation applied or skipped (vLLM's
     * {@code use_activation}); {@code false} answers the raw head output, which is not normalized.
     */
    private ClassifierOutput[] classifyBatch(LoraSelection lora, int[][] inputs, boolean activation) {
        try {
            return service.classifyOutputsAll(lora, inputs, activation);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the classify forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** True when a sequence classifier without a decision head answers {@code /classify}. */
    private void requireSequenceClassifier() {
        if (service.decisionHead()) {
            throw ServerJson.badRequest(service.architecture() + " is a decision model; its decision scores come"
                    + " from /pooling task classify");
        }
        if (service.sequenceClassifier()) {
            return;
        }
        if (service.tokenClassifier()) {
            throw ServerJson.badRequest("classification needs a pooled classifier; " + service.architecture()
                    + " classifies per token (task token_classify)");
        }
        throw ServerJson.badRequest("classification needs a RANK classifier or another sequence classifier; "
                + service.architecture() + " pools by " + service.poolingKind() + " and stores no classifier head");
    }

    /**
     * vLLM's generic pooling endpoint: one pooling task ({@code embed}, the model's default, or {@code
     * token_embed}, {@code token_classify}, {@code classify}) over the completion or chat input forms, each entry
     * answering {@code data} as a vector, per-token vectors or per-token scores, or a base64 string.
     */
    private Response poolingOf(JsonView body) {
        String requestId = poolingRequestFields(body);
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        if (body.raw("dimensions") != null) {
            throw ServerJson.badRequest("dimensions is currently not supported");
        }
        String task = poolingTask(body);
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength(),
                service.sparsePooling() != null);
        if (service.sparsePooling() != null && task.equals("embed") && options.base64()) {
            throw ServerJson.badRequest("encoding_format base64 does not apply to a sparse embedding; the"
                    + " token_classify-sparse response is a {token_id: weight} object");
        }
        List<PoolingInputs.Input> inputs = PoolingInputs.of(body, service, options);
        long tokens = 0;
        for (PoolingInputs.Input input : inputs) {
            if ((input.isImage() || input.isMedia()) && !task.equals("embed")) {
                throw ServerJson.badRequest(input.field() + " holds an image or audio part; task '" + task
                        + "' reads text inputs only");
            }
            tokens += input.isImage() || input.isMedia() ? 0 : input.tokens().length;
        }
        // every task's answers arrive as one batched request, in input order
        StepFilter filter = StepFilter.of(body, service, task);
        boolean combined = task.equals("embed&token_classify");
        Embedded[] embedded = task.equals("embed") || combined ? embeddedAll(lora, inputs) : null;
        float[][][] tokenRows = null;
        ClassifierOutput[] outputs = null;
        float[][] scores = null;
        if (task.equals("token_embed")) {
            tokenRows = embedTokenRows(lora, textTokens(inputs));
        } else if (task.equals("token_classify") || combined) {
            int[][] batch = textTokens(inputs);
            float[][][] rows = tokenClassifyRows(lora, batch);
            tokenRows = filter == null ? rows : StepFilter.apply(rows, batch, filter);
        } else if (task.equals("classify")) {
            int[][] batch = textTokens(inputs);
            if (service.decisionHead()) {
                scores = classifyScoresBatch(lora, batch);
            } else {
                outputs = classifyBatch(lora, batch);
            }
        }
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            Object value;
            switch (task) {
                case "embed" -> value = service.sparsePooling() != null
                        ? sparseVocabulary(embedded[i].vector())
                        : options.encoded(options.shaped(embedded[i].vector()));
                case "embed&token_classify" ->
                        value = options.encoded(combinedVector(embedded[i].vector(), tokenRows[i]));
                case "token_classify" ->
                        value = tokenClassifyData(service.tokenScoreProjection(), inputs.get(i).tokens(),
                                tokenRows[i], options);
                case "token_embed" -> value = tokenEmbeddingData(tokenRows[i], options);
                default -> value = options.encoded(service.decisionHead() ? scores[i] : classScores(outputs[i]));
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("object", "pooling");
            entry.put("data", value);
            data.add(entry);
        }
        return listEnvelope("pooling-", requestId, model, data, tokens);
    }

    /** The token arrays of the text inputs, in input order (the task's callers read text only). */
    private int[][] textTokens(List<PoolingInputs.Input> inputs) {
        List<int[]> batch = new ArrayList<>();
        for (PoolingInputs.Input input : inputs) {
            if (!input.isImage()) {
                batch.add(input.tokens());
            }
        }
        return batch.toArray(new int[0][]);
    }

    /** Per-token classifier rows of every input, one batched request. */
    private float[][][] tokenClassifyRows(LoraSelection lora, int[][] inputs) {
        try {
            return service.tokenClassifyAll(lora, inputs);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the token classification forward failed: " + e.getMessage(),
                    "server_error");
        }
    }

    /**
     * vLLM's process-reward step filter (qwen2_rm.py:77-88): {@code step_tag_id} keeps only the tokens carrying
     * that id, {@code returned_token_ids} then picks the classifier outputs those tokens answer. Either field
     * alone applies its half; both apply only to the {@code token_classify} task over a per-token classifier and
     * refuse by name everywhere else.
     */
    private record StepFilter(int stepTagId, boolean byTag, int[] returnedIds) {

        /** The filter of {@code body} when it names {@code step_tag_id} or {@code returned_token_ids}, else null. */
        static StepFilter of(JsonView body, EngineService service, String task) {
            boolean byTag = body.raw("step_tag_id") != null;
            boolean byIds = body.raw("returned_token_ids") != null;
            // vLLM merges the pooler config's step_tag_id into a request that omits it (PoolingParams
            // ._merge_default_parameters), so a bare token_classify on the process-reward model filters by it
            boolean byDefault = !byTag && "token_classify".equals(task) && service.tokenClassifier()
                    && service.defaultStepTagId() >= 0;
            if (!byTag && !byIds && !byDefault) {
                return null;
            }
            if (!"token_classify".equals(task)) {
                throw ServerJson.badRequest((byTag ? "step_tag_id" : "returned_token_ids")
                        + " applies only to the token_classify task, not '" + task + "'");
            }
            if (service.sparsePooling() != null) {
                throw ServerJson.badRequest("step_tag_id and returned_token_ids apply to the token_classify task;"
                        + " a sparse model reduces its classifier scores into the embed output and offers no"
                        + " per-token rows to filter");
            }
            if (!service.tokenClassifier()) {
                throw ServerJson.badRequest("step_tag_id and returned_token_ids apply to the token_classify task"
                        + " over a per-token classifier (vLLM's STEP token pooling); " + service.architecture()
                        + " (" + service.supportClass() + ", pooling " + service.poolingKind() + ") is not one");
            }
            if (byTag && body.optI32("step_tag_id", -1) < 0) {
                throw ServerJson.badRequest("step_tag_id must be a token id, got " + body.optI32("step_tag_id", -1));
            }
            int[] returnedIds = byIds ? OpenAiRequests.tokens(listField(body, "returned_token_ids"),
                    Integer.MAX_VALUE, "returned_token_ids") : null;
            int stepTagId = byTag ? body.optI32("step_tag_id", -1) : service.defaultStepTagId();
            return new StepFilter(stepTagId, byTag || byDefault, returnedIds);
        }

        /** The filtered rows of every input, in input order. */
        static float[][][] apply(float[][][] rows, int[][] tokens, StepFilter filter) {
            int width = 0;
            for (float[][] inputRows : rows) {
                if (inputRows.length > 0) {
                    width = inputRows[0].length;
                    break;
                }
            }
            if (filter.returnedIds != null) {
                for (int id : filter.returnedIds) {
                    if (id >= width) {
                        throw ServerJson.badRequest("returned_token_ids entry " + id
                                + " is outside the " + width + " classifier outputs");
                    }
                }
            }
            float[][][] out = new float[rows.length][][];
            for (int i = 0; i < out.length; i++) {
                out[i] = filter.one(rows[i], tokens[i], width);
            }
            return out;
        }

        /** The kept rows of one input: the {@code step_tag_id} tokens, their {@code returned_token_ids} columns. */
        private float[][] one(float[][] rows, int[] ids, int width) {
            List<float[]> kept = new ArrayList<>();
            for (int t = 0; t < rows.length; t++) {
                if (!byTag || ids[t] == stepTagId) {
                    kept.add(rows[t]);
                }
            }
            if (returnedIds == null) {
                return kept.toArray(new float[0][]);
            }
            float[][] picked = new float[kept.size()][returnedIds.length];
            for (int k = 0; k < kept.size(); k++) {
                for (int j = 0; j < returnedIds.length; j++) {
                    picked[k][j] = kept.get(k)[returnedIds[j]];
                }
            }
            return picked;
        }
    }

    /** Per-token embeddings of every input, one batched request. */
    private float[][][] embedTokenRows(LoraSelection lora, int[][] inputs) {
        try {
            return service.embedTokensAll(lora, inputs);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** The decision scores of every input, one batched request. */
    private float[][] classifyScoresBatch(LoraSelection lora, int[][] inputs) {
        try {
            return service.classifyScoresAll(lora, inputs);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the classify forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** The pooling task of {@code body}, or the model's own task when unset. */
    private String poolingTask(JsonView body) {
        String asked = body.optStr("task", null);
        removedPoolingTask(body);
        boolean classify = service.decisionHead() || service.sequenceClassifier();
        boolean tokenClassify = service.tokenClassifier();
        boolean tokenClassifyTask = service.canTokenClassify();
        boolean perToken = service.perTokenEmbeddings();
        boolean sparse = service.sparsePooling() != null;
        if (asked == null) {
            // a sparse model answers as the embed output (vLLM's SPLADESparsePooler), its classifier scores
            // reducing into one vector per input; a multi-output model's default is its pooled dense embedding
            if (sparse) {
                return "embed";
            }
            if (service.tokenEmbeddingProjection() || service.tokenScoreProjection()) {
                return "embed";
            }
            return classify ? "classify" : tokenClassify ? "token_classify" : perToken ? "token_embed" : "embed";
        }
        switch (asked) {
            case "embed":
                if (sparse) {
                    return "embed";
                }
                if (classify || tokenClassifyTask && !service.tokenScoreProjection()) {
                    throw ServerJson.badRequest("task 'embed' needs an embedding model; " + service.architecture()
                            + " pools by " + service.poolingKind()
                            + (tokenClassify ? " and classifies per token" : " and classifies"));
                }
                if (perToken && !service.tokenEmbeddingProjection()) {
                    throw ServerJson.badRequest("task 'embed' needs a pooled embedding; " + service.architecture()
                            + " pools per token (none)");
                }
                return "embed";
            case "token_embed":
                if (!perToken && !service.tokenEmbeddingProjection()) {
                    throw ServerJson.badRequest("task 'token_embed' needs per-token embeddings (none pooling and no"
                            + " classifier); " + service.architecture() + " (" + service.supportClass()
                            + ", pooling " + service.poolingKind() + ") has none");
                }
                return "token_embed";
            case "token_classify":
                if (sparse) {
                    throw ServerJson.badRequest("task 'token_classify' scores a sparse model's classifier into the"
                            + " embed output; " + service.architecture() + " ("
                            + service.supportClass() + ", sparse " + service.sparsePooling()
                            + ") offers no per-token rows");
                }
                if (!tokenClassifyTask) {
                    throw ServerJson.badRequest("task 'token_classify' needs a classifier head; "
                            + service.architecture() + " (" + service.supportClass() + ", pooling "
                            + service.poolingKind() + ") stores none");
                }
                return "token_classify";
            case "classify":
                if (!classify) {
                    throw ServerJson.badRequest("task 'classify' needs a RANK classifier, another sequence"
                            + " classifier or a decision head; " + service.architecture() + " pools by "
                            + service.poolingKind()
                            + (tokenClassify ? " and classifies per token; use token_classify" : ""));
                }
                return "classify";
            case "embed&token_classify":
                if (!service.tokenScoreProjection() || service.poolingKind() == com.eonmux.neuralfree.gguf.arch
                        .PoolingKind.NONE) {
                    throw ServerJson.badRequest("task 'embed&token_classify' needs a model with both a dense embedding"
                            + " and a per-token sparse classifier (vLLM's Bge-M3 dispatch); " + service.architecture()
                            + " (" + service.supportClass() + ", pooling " + service.poolingKind() + ") offers "
                            + (sparse ? "only the sparse embedding"
                            : tokenClassify ? "only the per-token classifier" : "neither"));
                }
                return "embed&token_classify";
            default:
                throw ServerJson.badRequest("Unsupported task: '" + asked + "'. Supported tasks: [embed, classify,"
                        + " token_classify, token_embed"
                        + (service.tokenScoreProjection() ? ", embed&token_classify" : "") + "]");
        }
    }

    /** Per-token vectors as JSON rows, or one base64 string of their float32 bytes. */
    private Object tokenEmbeddingData(float[][] rows, EmbeddingOptions options) {
        if (options.base64()) {
            java.nio.ByteBuffer bytes = java.nio.ByteBuffer
                    .allocate(Float.BYTES * java.util.Arrays.stream(rows).mapToInt(r -> r.length).sum())
                    .order(options.order());
            for (float[] row : rows) {
                for (float x : row) {
                    bytes.putFloat(x);
                }
            }
            return java.util.Base64.getEncoder().encodeToString(bytes.array());
        }
        List<Object> out = new ArrayList<>();
        for (float[] row : rows) {
            out.add(EmbeddingOptions.values(row));
        }
        return out;
    }

    /**
     * The per-token classify task's data: a model with a separate per-token score projection answers vLLM's sparse
     * shape, a {@code {token_id: weight}} object pairing each kept prompt token with its score (repeated ids keep
     * their largest weight); every other token classifier answers the nested score rows.
     */
    private Object tokenClassifyData(boolean sparseShape, int[] tokens, float[][] rows, EmbeddingOptions options) {
        if (!sparseShape || rows.length == 0 || rows[0].length != 1) {
            return tokenEmbeddingData(rows, options);
        }
        return sparseWeights(tokens, rows);
    }

    /**
     * The sparse {@code {token_id: weight}} map of BGE-M3's per-token scores: the boundary rows the head dropped are
     * paired off the prompt's ends, and a token id repeated in the prompt keeps its largest weight, as vLLM's sparse
     * post-processing does.
     */
    private Map<String, Object> sparseWeights(int[] tokens, float[][] rows) {
        int from = 0;
        int to = tokens.length;
        if (to > from && service.tokenBoundaryBos() >= 0 && tokens[0] == service.tokenBoundaryBos()) {
            from++;
        }
        if (to > from && service.tokenBoundaryEos() >= 0 && tokens[to - 1] == service.tokenBoundaryEos()) {
            to--;
        }
        if (to - from != rows.length) {
            throw ServerJson.error(500, "the per-token scores do not line up with the prompt's tokens ("
                    + rows.length + " rows for " + (to - from) + " tokens)", "server_error");
        }
        Map<String, Object> weights = new LinkedHashMap<>();
        for (int t = from; t < to; t++) {
            String id = Integer.toString(tokens[t]);
            Object previous = weights.get(id);
            double weight = rows[t - from][0];
            if (previous == null || weight > (Double) previous) {
                weights.put(id, weight);
            }
        }
        return weights;
    }

    /**
     * The sparse {@code {token_id: weight}} object of a SPLADE vocabulary vector: the vocabulary entries whose
     * weight is non-zero, the token id being the vocabulary index.
     */
    private static Map<String, Object> sparseVocabulary(float[] weights) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int id = 0; id < weights.length; id++) {
            if (weights[id] != 0.0f) {
                out.put(Integer.toString(id), (double) weights[id]);
            }
        }
        return out;
    }

    /** The combined {@code embed&token_classify} vector: the pooled embedding then the flattened token scores. */
    private static float[] combinedVector(float[] pooled, float[][] rows) {
        int width = rows.length == 0 ? 0 : rows[0].length;
        float[] out = new float[pooled.length + rows.length * width];
        System.arraycopy(pooled, 0, out, 0, pooled.length);
        for (int t = 0; t < rows.length; t++) {
            System.arraycopy(rows[t], 0, out, pooled.length + t * width, width);
        }
        return out;
    }

    /** vLLM's generative scoring: the next-token probability of the first label token after query and item. */
    private Response generativeScoringOf(JsonView body) {
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        if (service.supportClass() != com.eonmux.neuralfree.gguf.arch.SupportClass.GEN) {
            throw ServerJson.badRequest("generative scoring needs a generative model; " + service.architecture()
                    + " is a " + service.supportClass() + " model");
        }
        int[] query = generativeSide(body.raw("query"), "query", body.optBool("add_special_tokens", true));
        if (!(body.raw("items") instanceof List<?> items) || items.isEmpty()) {
            throw ServerJson.badRequest("items must contain at least one item.");
        }
        int[] labels = OpenAiRequests.tokens(listField(body, "label_token_ids"), service.vocabSize(),
                "label_token_ids");
        if (labels.length == 0) {
            throw ServerJson.badRequest("label_token_ids must contain at least one token ID.");
        }
        boolean applySoftmax = body.optBool("apply_softmax", true);
        boolean itemFirst = body.optBool("item_first", false);
        int cap = service.maxModelLength() - 1;
        List<Map<String, Object>> data = new ArrayList<>();
        long promptTokens = 0;
        for (int i = 0; i < items.size(); i++) {
            int[] item = generativeSide(items.get(i), "items[" + i + "]", false);
            int[] prompt = join(itemFirst ? item : query, itemFirst ? query : item, cap);
            float[] logits;
            try {
                logits = service.nextLogits(lora, prompt);
            } catch (UnsupportedOperationException e) {
                throw ServerJson.badRequest(e.getMessage());
            } catch (IOException e) {
                throw ServerJson.error(500, "the scoring forward failed: " + e.getMessage(), "server_error");
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("object", "score");
            entry.put("score", labelScore(logits, labels, applySoftmax));
            data.add(entry);
            promptTokens += prompt.length;
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", (long) items.size());
        usage.put("total_tokens", promptTokens + items.size());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "generative-scoring-" + System.nanoTime());
        out.put("object", "list");
        out.put("created", System.currentTimeMillis() / 1000);
        out.put("model", model);
        out.put("data", data);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** One generative-scoring side: a string, or an array of token ids in the vocabulary. */
    private int[] generativeSide(Object raw, String field, boolean addSpecial) {
        if (raw instanceof String s) {
            return service.encode(s, addSpecial, true);
        }
        if (raw instanceof List<?> ids && !ids.isEmpty()) {
            return OpenAiRequests.tokens(ids, service.vocabSize(), field);
        }
        throw ServerJson.badRequest(field + " must be a string or an array of token ids");
    }

    /** A required integer array member of {@code body}. */
    private static List<?> listField(JsonView body, String field) {
        if (!(body.raw(field) instanceof List<?> list)) {
            throw ServerJson.badRequest(field + " must be an array of token ids");
        }
        return list;
    }

    /** Both parts concatenated, at most {@code cap} tokens, the first kept. */
    private int[] join(int[] head, int[] tail, int cap) {
        int n = Math.min(head.length + tail.length, Math.max(0, cap));
        int[] out = new int[n];
        System.arraycopy(head, 0, out, 0, Math.min(head.length, n));
        int fill = Math.min(tail.length, Math.max(0, n - head.length));
        if (fill > 0) {
            System.arraycopy(tail, 0, out, head.length, fill);
        }
        return out;
    }

    /** The probability of {@code labels[0]}: softmax over the labels, or over the whole vocabulary. */
    private static double labelScore(float[] logits, int[] labels, boolean applySoftmax) {
        if (applySoftmax) {
            double max = Double.NEGATIVE_INFINITY;
            for (int id : labels) {
                max = Math.max(max, logits[id]);
            }
            double sum = 0;
            for (int id : labels) {
                sum += Math.exp(logits[id] - max);
            }
            return Math.exp(logits[labels[0]] - max) / sum;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float logit : logits) {
            max = Math.max(max, logit);
        }
        double sum = 0;
        for (float logit : logits) {
            sum += Math.exp(logit - max);
        }
        return Math.exp(logits[labels[0]] - max) / sum;
    }

    /**
     * vLLM's list envelope: id, object, created, model, data and the prompt-token usage. The id carries the
     * request's {@code request_id} when it names one, as vLLM echoes it.
     */
    private Response listEnvelope(String idPrefix, String requestId, String model, List<Map<String, Object>> data,
                                   long tokens) {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", tokens);
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", idPrefix + (requestId != null ? requestId : Long.toString(System.nanoTime())));
        out.put("object", "list");
        out.put("created", System.currentTimeMillis() / 1000);
        out.put("model", model);
        out.put("data", data);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The vLLM pooling-request fields every endpoint shares beside its payload: the scheduling {@code priority}
     * (any value but 0 refuses under the fcfs policy, as the batch engine refuses at submit), the {@code
     * cache_salt} validation, and the refusals of {@code padding} beyond {@code do_not_pad} and a non-empty
     * {@code mm_processor_kwargs}, whose engine effects this server does not run. Returns the request's response
     * id: the {@code X-Request-Id} header first, then the body {@code request_id}, as vLLM names it.
     */
    private String poolingRequestFields(JsonView body) {
        String refusal = service.schedulingPolicy().priorityRefusal(OpenAiRequests.priority(body));
        if (refusal != null) {
            throw ServerJson.badRequest(refusal);
        }
        OpenAiRequests.cacheSalt(body);
        Object padding = body.raw("padding");
        if (padding != null) {
            if (!(padding instanceof String p) || (!p.equals("do_not_pad") && !p.equals("max_length"))) {
                throw ServerJson.badRequest("padding must be max_length or do_not_pad, got " + body.raw("padding"));
            }
            if (p.equals("max_length")) {
                throw ServerJson.badRequest("padding 'max_length' is not supported: the pooling forward runs no"
                        + " attention mask, so pad tokens would enter every attention row and change the pooled"
                        + " result; prompts pool at their own length");
            }
        }
        Object processor = body.raw("mm_processor_kwargs");
        if (processor != null && !(processor instanceof Map<?, ?> m && m.isEmpty())) {
            throw ServerJson.badRequest("mm_processor_kwargs: per-request processor options are not supported;"
                    + " the media processor is the server's");
        }
        return OpenAiRequests.requestId(http, body);
    }

    /**
     * The removed pooling tasks, refused with vLLM's messages where its request validator rejects them: on the
     * embedding, classification and pooling requests, not on score and rerank, whose protocol has no such check.
     */
    private static void removedPoolingTask(JsonView body) {
        Object task = body.raw("task");
        if ("score".equals(task)) {
            throw ServerJson.badRequest("`score` task was removed; use `classify` instead.");
        }
        if ("encode".equals(task)) {
            throw ServerJson.badRequest("`encode` task was removed; use `token_embed` or `token_classify`"
                    + " instead.");
        }
    }

    /** The per-input pooled vector of one text input under {@code lora}, a refusal or failure mapped to a response. */
    private float[] pooled(LoraSelection lora, int[] ids) {
        try {
            return service.embed(lora, ids);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    /**
     * vLLM's rerank: the {@code query} and every document are {@link ScoreSides} entries (a string, an object with
     * a {@code content} array, or a Cohere {@code {"text": ...}} document); a RANK classifier scores each text pair
     * (cross-encoder), a per-token embedding model the MaxSim of the two token-row sets (late interaction), a
     * pooled embedding model the cosine similarity of the two embeddings (bi-encoder, a CLIP dual
     * encoder's image entries included; a SigLIP dual encoder's sigmoid score of it). The results come best first, {@code top_n} of them, each with its
     * document when {@code return_documents} is set: {@code {"text"}} or vLLM's {@code {"multi_modal"}} parts.
     */
    private Response rerankOf(JsonView body) {
        String requestId = poolingRequestFields(body);
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        boolean activation = OpenAiRequests.useActivation(body);
        if (body.raw("query") == null) {
            throw ServerJson.badRequest("query is required");
        }
        ScoreSides.Entry query = ScoreSides.entry(body.raw("query"), "query");
        if (query == null) {
            throw ServerJson.badRequest("query must be a string or an object with a content array");
        }
        Object rawDocuments = body.raw("documents");
        if (rawDocuments == null) {
            throw ServerJson.badRequest("documents is required and must be an array");
        }
        List<ScoreSides.Entry> documents = ScoreSides.side(rawDocuments, "documents");
        int topN = topN(body, documents.size());
        boolean returnText = body.optBool("return_documents", false);
        ScoringOptions options = ScoringOptions.of(body, service);
        refuseMediaTruncation(List.of(query), documents, options);
        ScoringMode mode = scoringMode("rerank");
        String instruction = instructionOf(options, mode);
        double[] scoredScores = new double[documents.size()];
        long tokens = 0;
        if (mode == ScoringMode.LATE) {
            float[][] queryRows = sideTokenRows(lora, List.of(query), options.maxQuery())[0];
            float[][][] documentRows = sideTokenRows(lora, documents, options.maxDoc());
            for (int i = 0; i < documents.size(); i++) {
                tokens += sideTokens(query, options.maxQuery()) + sideTokens(documents.get(i), options.maxDoc());
                scoredScores[i] = LateInteraction.maxSim(queryRows, documentRows[i]);
            }
        } else if (mode == ScoringMode.CROSS) {
            // the text pairs score as one batched rerank request; a pair with media parts scores alone through
            // the projector (the batched request takes token inputs only)
            List<int[]> batch = new ArrayList<>();
            List<Integer> textAt = new ArrayList<>();
            List<Integer> mediaAt = new ArrayList<>();
            for (int i = 0; i < documents.size(); i++) {
                ScoreSides.Entry document = documents.get(i);
                if (query.isMedia() || document.isMedia()) {
                    mediaAt.add(i);
                } else {
                    batch.add(crossInput(query, document, options));
                    textAt.add(i);
                }
            }
            float[] answers = crossScores(lora, batch, activation);
            for (int b = 0; b < answers.length; b++) {
                scoredScores[textAt.get(b)] = answers[b];
                tokens += batch.get(b).length;
            }
            for (int i : mediaAt) {
                PairScore pair = mediaCrossScore(lora, query, documents.get(i), options, activation);
                scoredScores[i] = pair.score();
                tokens += pair.tokens();
            }
        } else {
            float[] queryVector = sideVectors(lora, List.of(query), options.maxQuery())[0];
            float[][] documentVectors = sideVectors(lora, documents, options.maxDoc());
            for (int i = 0; i < documents.size(); i++) {
                tokens += sideTokens(query, options.maxQuery()) + sideTokens(documents.get(i), options.maxDoc());
                scoredScores[i] = service.biEncoderScore(cosine(queryVector, documentVectors[i]));
            }
        }
        List<double[]> scored = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            scored.add(new double[]{scoredScores[i], i});
        }
        scored.sort((a, b) -> Double.compare(b[0], a[0]));
        List<Map<String, Object>> results = new ArrayList<>();
        for (int r = 0; r < Math.min(topN, scored.size()); r++) {
            int index = (int) scored.get(r)[1];
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("index", index);
            result.put("relevance_score", scored.get(r)[0]);
            if (returnText) {
                ScoreSides.Entry document = documents.get(index);
                result.put("document", document.content() == null ? Map.of("text", document.text())
                        : Map.of("multi_modal", document.content()));
            }
            results.add(result);
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", tokens);
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "rerank-" + (requestId != null ? requestId : Long.toString(System.nanoTime())));
        out.put("model", model);
        out.put("results", results);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** vLLM's {@code top_n}: 0 serves every document; a negative value refuses by name. */
    private static int topN(JsonView body, int count) {
        int n = body.optI32("top_n", 0);
        if (n < 0) {
            throw ServerJson.badRequest("top_n must be 0 (every document) or positive, got " + n);
        }
        return n == 0 ? count : n;
    }

    /**
     * Media sides keep their rows, so the pair truncation refuses beside them, as the media pooling path refuses
     * it; a text-only pair cuts.
     */
    private static void refuseMediaTruncation(List<ScoreSides.Entry> one, List<ScoreSides.Entry> two,
                                              ScoringOptions options) {
        if (options.truncation().truncate() == 0) {
            return;
        }
        for (List<ScoreSides.Entry> side : List.of(one, two)) {
            for (ScoreSides.Entry entry : side) {
                if (entry.isMedia()) {
                    throw ServerJson.badRequest(entry.field() + " holds an image or audio part, but"
                            + " truncate_prompt_tokens cuts text tokens only");
                }
            }
        }
    }
}
