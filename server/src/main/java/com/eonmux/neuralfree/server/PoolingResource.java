package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
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

    /** The score and rerank scoring over this resource's service ({@link PoolingScoring}). */
    final PoolingScoring scoring = new PoolingScoring(this);

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
        return OpenAiResource.mapped(() -> target.scoring.rerankOf(body));
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
        return OpenAiResource.mapped(() -> target.scoring.scoreOf(body));
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
        if (service.pooling().perTokenEmbeddings() && service.pooling().sparsePooling() == null
                && service.pooling().poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            throw ServerJson.badRequest("Pooling type 'none' is not OAI compatible. Please use a different pooling"
                    + " type, or the native /embeddings endpoint");
        }
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength(),
                service.pooling().sparsePooling() != null);
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
            pooled = service.pooling().embedAll(lora, batch.toArray(new int[0][]));
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
                return new Embedded(service.pooling().embedMedia(lora, prompt), prompt.nTokens());
            } catch (UnsupportedOperationException e) {
                throw ServerJson.badRequest(e.getMessage());
            } catch (IOException e) {
                throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
            }
        }
        return new Embedded(pooled(lora, input.tokens()), input.tokens().length);
    }

    /** The dual encoder's embedding of one image under {@code lora}, a refusal naming it mapped to a 400. */
    float[] image(LoraSelection lora, String reference, String field) {
        try {
            return service.pooling().embedImage(lora, reference, field);
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
                service.pooling().sparsePooling() != null);
        boolean perToken = service.pooling().perTokenEmbeddings() && service.pooling().sparsePooling() == null
                && service.pooling().poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE;
        List<int[]> prompts = LlamaPrompts.of(raw, service);
        int[][] truncated = new int[prompts.size()][];
        for (int i = 0; i < prompts.size(); i++) {
            truncated[i] = options.truncated(prompts.get(i));
        }
        float[][] pooled = null;
        float[][][] rows = null;
        try {
            if (perToken) {
                rows = service.pooling().embedTokensAll(lora, truncated);
            } else {
                pooled = service.pooling().embedAll(lora, truncated);
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
            return service.pooling().classifyOutputsAll(lora, inputs, activation);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the classify forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** True when a sequence classifier without a decision head answers {@code /classify}. */
    private void requireSequenceClassifier() {
        if (service.pooling().decisionHead()) {
            throw ServerJson.badRequest(service.architecture() + " is a decision model; its decision scores come"
                    + " from /pooling task classify");
        }
        if (service.pooling().sequenceClassifier()) {
            return;
        }
        if (service.pooling().tokenClassifier()) {
            throw ServerJson.badRequest("classification needs a pooled classifier; " + service.architecture()
                    + " classifies per token (task token_classify)");
        }
        throw ServerJson.badRequest("classification needs a RANK classifier or another sequence classifier; "
                + service.architecture() + " pools by " + service.pooling().poolingKind() + " and stores no classifier head");
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
                service.pooling().sparsePooling() != null);
        if (service.pooling().sparsePooling() != null && task.equals("embed") && options.base64()) {
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
            if (service.pooling().decisionHead()) {
                scores = classifyScoresBatch(lora, batch);
            } else {
                outputs = classifyBatch(lora, batch);
            }
        }
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            Object value;
            switch (task) {
                case "embed" -> value = service.pooling().sparsePooling() != null
                        ? sparseVocabulary(embedded[i].vector())
                        : options.encoded(options.shaped(embedded[i].vector()));
                case "embed&token_classify" ->
                        value = options.encoded(combinedVector(embedded[i].vector(), tokenRows[i]));
                case "token_classify" ->
                        value = tokenClassifyData(service.pooling().tokenScoreProjection(), inputs.get(i).tokens(),
                                tokenRows[i], options);
                case "token_embed" -> value = tokenEmbeddingData(tokenRows[i], options);
                default -> value = options.encoded(service.pooling().decisionHead() ? scores[i] : classScores(outputs[i]));
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
            return service.pooling().tokenClassifyAll(lora, inputs);
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
            boolean byDefault = !byTag && "token_classify".equals(task) && service.pooling().tokenClassifier()
                    && service.pooling().defaultStepTagId() >= 0;
            if (!byTag && !byIds && !byDefault) {
                return null;
            }
            if (!"token_classify".equals(task)) {
                throw ServerJson.badRequest((byTag ? "step_tag_id" : "returned_token_ids")
                        + " applies only to the token_classify task, not '" + task + "'");
            }
            if (service.pooling().sparsePooling() != null) {
                throw ServerJson.badRequest("step_tag_id and returned_token_ids apply to the token_classify task;"
                        + " a sparse model reduces its classifier scores into the embed output and offers no"
                        + " per-token rows to filter");
            }
            if (!service.pooling().tokenClassifier()) {
                throw ServerJson.badRequest("step_tag_id and returned_token_ids apply to the token_classify task"
                        + " over a per-token classifier (vLLM's STEP token pooling); " + service.architecture()
                        + " (" + service.pooling().supportClass() + ", pooling " + service.pooling().poolingKind() + ") is not one");
            }
            if (byTag && body.optI32("step_tag_id", -1) < 0) {
                throw ServerJson.badRequest("step_tag_id must be a token id, got " + body.optI32("step_tag_id", -1));
            }
            int[] returnedIds = byIds ? OpenAiRequests.tokens(listField(body, "returned_token_ids"),
                    Integer.MAX_VALUE, "returned_token_ids") : null;
            int stepTagId = byTag ? body.optI32("step_tag_id", -1) : service.pooling().defaultStepTagId();
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
            return service.pooling().embedTokensAll(lora, inputs);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** The decision scores of every input, one batched request. */
    private float[][] classifyScoresBatch(LoraSelection lora, int[][] inputs) {
        try {
            return service.pooling().classifyScoresAll(lora, inputs);
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
        boolean classify = service.pooling().decisionHead() || service.pooling().sequenceClassifier();
        boolean tokenClassify = service.pooling().tokenClassifier();
        boolean tokenClassifyTask = service.pooling().canTokenClassify();
        boolean perToken = service.pooling().perTokenEmbeddings();
        boolean sparse = service.pooling().sparsePooling() != null;
        if (asked == null) {
            // a sparse model answers as the embed output (vLLM's SPLADESparsePooler), its classifier scores
            // reducing into one vector per input; a multi-output model's default is its pooled dense embedding
            if (sparse) {
                return "embed";
            }
            if (service.pooling().tokenEmbeddingProjection() || service.pooling().tokenScoreProjection()) {
                return "embed";
            }
            return classify ? "classify" : tokenClassify ? "token_classify" : perToken ? "token_embed" : "embed";
        }
        switch (asked) {
            case "embed":
                if (sparse) {
                    return "embed";
                }
                if (classify || tokenClassifyTask && !service.pooling().tokenScoreProjection()) {
                    throw ServerJson.badRequest("task 'embed' needs an embedding model; " + service.architecture()
                            + " pools by " + service.pooling().poolingKind()
                            + (tokenClassify ? " and classifies per token" : " and classifies"));
                }
                if (perToken && !service.pooling().tokenEmbeddingProjection()) {
                    throw ServerJson.badRequest("task 'embed' needs a pooled embedding; " + service.architecture()
                            + " pools per token (none)");
                }
                return "embed";
            case "token_embed":
                if (!perToken && !service.pooling().tokenEmbeddingProjection()) {
                    throw ServerJson.badRequest("task 'token_embed' needs per-token embeddings (none pooling and no"
                            + " classifier); " + service.architecture() + " (" + service.pooling().supportClass()
                            + ", pooling " + service.pooling().poolingKind() + ") has none");
                }
                return "token_embed";
            case "token_classify":
                if (sparse) {
                    throw ServerJson.badRequest("task 'token_classify' scores a sparse model's classifier into the"
                            + " embed output; " + service.architecture() + " ("
                            + service.pooling().supportClass() + ", sparse " + service.pooling().sparsePooling()
                            + ") offers no per-token rows");
                }
                if (!tokenClassifyTask) {
                    throw ServerJson.badRequest("task 'token_classify' needs a classifier head; "
                            + service.architecture() + " (" + service.pooling().supportClass() + ", pooling "
                            + service.pooling().poolingKind() + ") stores none");
                }
                return "token_classify";
            case "classify":
                if (!classify) {
                    throw ServerJson.badRequest("task 'classify' needs a RANK classifier, another sequence"
                            + " classifier or a decision head; " + service.architecture() + " pools by "
                            + service.pooling().poolingKind()
                            + (tokenClassify ? " and classifies per token; use token_classify" : ""));
                }
                return "classify";
            case "embed&token_classify":
                if (!service.pooling().tokenScoreProjection() || service.pooling().poolingKind() == com.eonmux.neuralfree.gguf.arch
                        .PoolingKind.NONE) {
                    throw ServerJson.badRequest("task 'embed&token_classify' needs a model with both a dense embedding"
                            + " and a per-token sparse classifier (vLLM's Bge-M3 dispatch); " + service.architecture()
                            + " (" + service.pooling().supportClass() + ", pooling " + service.pooling().poolingKind() + ") offers "
                            + (sparse ? "only the sparse embedding"
                            : tokenClassify ? "only the per-token classifier" : "neither"));
                }
                return "embed&token_classify";
            default:
                throw ServerJson.badRequest("Unsupported task: '" + asked + "'. Supported tasks: [embed, classify,"
                        + " token_classify, token_embed"
                        + (service.pooling().tokenScoreProjection() ? ", embed&token_classify" : "") + "]");
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
        if (to > from && service.pooling().tokenBoundaryBos() >= 0 && tokens[0] == service.pooling().tokenBoundaryBos()) {
            from++;
        }
        if (to > from && service.pooling().tokenBoundaryEos() >= 0 && tokens[to - 1] == service.pooling().tokenBoundaryEos()) {
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
        if (service.pooling().supportClass() != com.eonmux.neuralfree.gguf.arch.SupportClass.GEN) {
            throw ServerJson.badRequest("generative scoring needs a generative model; " + service.architecture()
                    + " is a " + service.pooling().supportClass() + " model");
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
                logits = service.pooling().nextLogits(lora, prompt);
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
    Response listEnvelope(String idPrefix, String requestId, String model, List<Map<String, Object>> data,
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
    String poolingRequestFields(JsonView body) {
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
            return service.pooling().embed(lora, ids);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }



}
