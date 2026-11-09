package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.ops.heads.LateInteraction;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The score and rerank scoring of {@link PoolingResource} (vLLM's {@code /score} and {@code /v1/rerank}): one
 * scorer per request model — a RANK or other sequence classifier scores every composed pair (cross-encoder), a
 * per-token embedding model the MaxSim of the two L2-normalized token-row sets (late interaction), a pooled
 * embedding model the cosine similarity of the two vectors (bi-encoder, a CLIP dual encoder's image entries
 * included and a SigLIP dual encoder its sigmoid score of it) — with the request's task instruction, per-side and
 * pair truncation and the media pairs that score alone through the projector. The embedding forwards it needs run
 * through the resource ({@code embeddedOf}, the dual encoder's {@code image}), and the responses it builds go
 * through the resource's envelope and request-field validation.
 */
final class PoolingScoring {

    /** The resource whose service and response helpers this scoring serves with. */
    private final PoolingResource owner;

    PoolingScoring(PoolingResource owner) {
        this.owner = owner;
    }

    /**
     * vLLM's score forms: two sides named {@code queries} + {@code documents}, {@code queries} + {@code items},
     * {@code data_1} + {@code data_2} or {@code text_1} + {@code text_2}, each a {@link ScoreSides} side. A RANK
     * classifier scores every text pair (cross-encoder); a per-token embedding model the MaxSim of the two
     * L2-normalized token-row sets (late interaction); an embedding model answers the cosine similarity of the
     * pooled vectors (bi-encoder), an image entry embedded by a CLIP dual encoder's vision tower, and a SigLIP dual
     * encoder its sigmoid score of that cosine ({@link EnginePooling#biEncoderScore}). One query is scored against
     * every document; otherwise the sides must have the same length.
     */
    Response scoreOf(JsonView body) {
        String requestId = owner.poolingRequestFields(body);
        String model = owner.service.loras().model(body);
        LoraSelection lora = owner.service.loras().selection(body, model);
        boolean activation = OpenAiRequests.useActivation(body);
        ScoringOptions options = ScoringOptions.of(body, owner.service);
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
                scores[i] = owner.service.pooling().biEncoderScore(cosine(query, documents[i]));
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
        return owner.listEnvelope("score-", requestId, model, data, tokens);
    }

    /** The tokens of one text entry after {@code max} truncation (0 keeps all). */
    int[] scoreTokens(String text, int max) {
        int[] ids = owner.service.encode(text, false, false);
        return max > 0 && ids.length > max ? java.util.Arrays.copyOf(ids, max) : ids;
    }

    /** The prompt tokens one entry counts: its truncated text tokens, none for an image. */
    long sideTokens(ScoreSides.Entry entry, int max) {
        return entry.isImage() ? 0 : scoreTokens(entry.text(), max).length;
    }

    /**
     * The embedding of every entry under {@code lora}, in entry order: the text entries as one batch (the batch
     * engine runs their prompts in its steps), an image through the dual encoder's vision tower and a media
     * conversation through the projector, each on its own. A text entry truncates to {@code max} content tokens
     * first, as vLLM truncates a score side before pooling it.
     */
    float[][] sideVectors(LoraSelection lora, List<ScoreSides.Entry> entries, int max) {
        float[][] vectors = new float[entries.size()][];
        List<Integer> textAt = new ArrayList<>();
        List<int[]> batch = new ArrayList<>();
        for (int i = 0; i < entries.size(); i++) {
            ScoreSides.Entry entry = entries.get(i);
            if (entry.isImage()) {
                vectors[i] = owner.image(lora, entry.image(), entry.field());
                continue;
            }
            if (entry.isMedia() && !owner.service.pooling().dualEncoder()) {
                vectors[i] = mediaVector(lora, sideChat(entry));
                continue;
            }
            if (entry.isMedia()) {
                ScoreSides.requireClipForm(entry); // refuses unless the entry is text-only, which pools as text
            }
            int[] ids = owner.service.encode(entry.text(), true, true);
            if (ids.length == 0) {
                throw ServerJson.badRequest(entry.field() + ": Input content cannot be empty");
            }
            textAt.add(i);
            batch.add(cappedTextTokens(entry.text(), max));
        }
        float[][] pooled;
        try {
            pooled = owner.service.pooling().embedAll(lora, batch.toArray(new int[0][]));
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
    int[] cappedTextTokens(String text, int max) {
        int[] ids = owner.service.encode(text, true, true);
        if (ids.length == 0) {
            throw ServerJson.badRequest("Input content cannot be empty");
        }
        if (max <= 0) {
            return ids;
        }
        int[] content = owner.service.encode(text, false, false);
        return content.length <= max ? ids
                : owner.service.encode(owner.service.tokenizer().decode(java.util.Arrays.copyOf(content, max)), true, true);
    }

    /** The text of at most {@code max} content tokens: the first tokens kept, decoded when the cap cuts. */
    String truncatedText(String text, int max) {
        if (max <= 0) {
            return text;
        }
        int[] content = owner.service.encode(text, false, false);
        return content.length <= max ? text : owner.service.tokenizer().decode(java.util.Arrays.copyOf(content, max));
    }

    /**
     * The per-token rows of every text entry under {@code lora}, one batched request for the late-interaction
     * scorer: an image or media entry refuses by name (the scorer reads text only) and a text entry truncates to
     * {@code max} content tokens first, as vLLM truncates a score side.
     */
    float[][][] sideTokenRows(LoraSelection lora, List<ScoreSides.Entry> entries, int max) {
        List<int[]> batch = new ArrayList<>();
        for (ScoreSides.Entry entry : entries) {
            if (entry.isImage() || entry.isMedia()) {
                throw ServerJson.badRequest(entry.field() + " holds an image or audio part; the late-interaction"
                        + " model " + owner.service.architecture() + " reads text only");
            }
            batch.add(cappedTextTokens(entry.text(), max));
        }
        try {
            return owner.service.pooling().embedTokensAll(lora, batch.toArray(new int[0][]));
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    /** The pooled vector of one media conversation under {@code lora} ({@link PoolingResource#embeddedOf} over its prompt). */
    float[] mediaVector(LoraSelection lora, ChatRequest chat) {
        return owner.embeddedOf(lora, new PoolingInputs.Input(null, null, chat, "input")).vector();
    }

    /** A one-message user conversation whose content is the raw parts of {@code entry}. */
    ChatRequest sideChat(ScoreSides.Entry entry) {
        Map<String, Object> chat = new LinkedHashMap<>();
        List<Object> messages = new ArrayList<>();
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", entry.content());
        messages.add(message);
        chat.put("messages", messages);
        return OpenAiRequests.chat(JsonView.of(chat, entry.field()), owner.service.options());
    }

    /** The scorer of a score/rerank request, over the model the service holds. */
    enum ScoringMode {
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
    ScoringMode scoringMode(String endpoint) {
        if (owner.service.pooling().decisionHead()) {
            throw ServerJson.badRequest(owner.service.architecture() + " is a decision model; " + endpoint
                    + " needs a sequence classifier or an embedding model");
        }
        if (owner.service.pooling().perTokenEmbeddings() && owner.service.pooling().sparsePooling() == null
                && owner.service.pooling().poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            return ScoringMode.LATE;
        }
        if (owner.service.pooling().sequenceClassifier()) {
            return ScoringMode.CROSS;
        }
        if (owner.service.pooling().poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            throw ServerJson.badRequest(endpoint + " needs pooled embeddings, a sequence classifier or per-token"
                    + " embeddings; " + owner.service.architecture() + " (" + owner.service.pooling().supportClass() + ", pooling "
                    + owner.service.pooling().poolingKind() + ") stores none of them");
        }
        return ScoringMode.BI;
    }

    /**
     * The request's task instruction, refused when no score prompt of this model can place it: a bi-encoder and a
     * late-interaction model read none, and a cross-encoder needs the {@code tokenizer.chat_template.rerank} prompt
     * to hold an {@code {instruction}} placeholder beside {@code query} and {@code document}.
     */
    String instructionOf(ScoringOptions options, ScoringMode mode) {
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
        String template = owner.service.tokenizer().rerankTemplate();
        if (template == null) {
            throw ServerJson.badRequest("instruction needs the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE + " prompt to place it; "
                    + owner.service.architecture() + " has none");
        }
        if (template.indexOf(com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.RERANK_INSTRUCTION) < 0) {
            throw ServerJson.badRequest("instruction has no place in the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE
                    + " prompt; it applies through an {instruction} placeholder");
        }
        return instruction;
    }

    /** One cross-encoder score and the tokens of its pair input. */
    record PairScore(double score, long tokens) {
    }

    /**
     * The cross-encoder input of one text pair: the sides truncate per their caps and compose through the model's
     * rerank prompt with the instruction ({@link #pairInput}), then the pair truncation cuts; an image entry
     * refuses by name (a media pair scores alone through the projector, as {@link #mediaCrossScore}).
     */
    int[] crossInput(ScoreSides.Entry query, ScoreSides.Entry document, ScoringOptions options) {
        for (ScoreSides.Entry entry : new ScoreSides.Entry[]{query, document}) {
            if (entry.isImage()) {
                throw ServerJson.badRequest(entry.field() + " holds an image, but the RANK classifier "
                        + owner.service.architecture() + " reads text only; image scoring needs a CLIP dual encoder");
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
    float[] crossScores(LoraSelection lora, List<int[]> inputs, boolean activation) {
        try {
            return owner.service.pooling().rerankScores(lora, inputs.toArray(new int[0][]), activation);
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
    int[] pairInput(String query, String document, ScoringOptions options) {
        int[] input;
        if (owner.service.tokenizer().rerankTemplate() != null) {
            input = owner.service.pooling().rerankInput(truncatedText(query, options.maxQuery()),
                    truncatedText(document, options.maxDoc()), options.instruction());
        } else {
            input = owner.service.pooling().rerankInput(scoreTokens(query, options.maxQuery()),
                    scoreTokens(document, options.maxDoc()));
        }
        return options.truncation().truncated(input);
    }

    /** The cross-encoder score of one media pair, and the rows its prompt counted. */
    PairScore mediaCrossScore(LoraSelection lora, ScoreSides.Entry query, ScoreSides.Entry document,
                                      ScoringOptions options, boolean activation) {
        if (options.maxQuery() > 0 || options.maxDoc() > 0) {
            throw ServerJson.badRequest("max_tokens_per_query and max_tokens_per_doc truncate text entries only;"
                    + " a media entry keeps its rows");
        }
        if (!owner.service.media().enabled()) {
            ScoreSides.Entry media = query.isMedia() ? query : document;
            throw ServerJson.badRequest(media.field() + " holds an image or audio part, but the RANK classifier "
                    + owner.service.architecture() + " reads text only; media scoring needs a media projector ("
                    + MediaOptions.MMPROJ_PROPERTY + ")");
        }
        MediaPrompt pair = rerankMediaPrompt(query, document, options.instruction(),
                ChatPlan.mediaLoras(owner.service, lora));
        try {
            return new PairScore(owner.service.pooling().rerankScoreMedia(lora, pair, activation), pair.nTokens());
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
    MediaPrompt rerankMediaPrompt(ScoreSides.Entry query, ScoreSides.Entry document, String instruction,
                                  Supplier<LoraSlotKey> mediaLoras) {
        String template = owner.service.tokenizer().rerankTemplate();
        if (template == null) {
            throw ServerJson.badRequest("media scoring needs the model's "
                    + com.eonmux.neuralfree.gguf.GGUFTokenizer.KEY_RERANK_TEMPLATE
                    + " prompt to place the query and document markers; " + owner.service.architecture() + " has none");
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
        ChatRequest request = OpenAiRequests.chat(JsonView.of(chat, "score sides"), owner.service.options());
        List<ChatMedia.Item> items = ChatPlan.mediaItems(owner.service, request, mediaLoras);
        MediaPrompt prompt;
        try {
            prompt = owner.service.media().prompt(rendered, items, List.of(), -1, text -> owner.service.encode(text, false, true),
                    mediaLoras);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        int[] suffix = owner.service.tokenizer().rerankSuffix();
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
    static String inlineText(ScoreSides.Entry entry) {
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
    static double cosine(float[] a, float[] b) {
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

    /**
     * vLLM's rerank: the {@code query} and every document are {@link ScoreSides} entries (a string, an object with
     * a {@code content} array, or a Cohere {@code {"text": ...}} document); a RANK classifier scores each text pair
     * (cross-encoder), a per-token embedding model the MaxSim of the two token-row sets (late interaction), a
     * pooled embedding model the cosine similarity of the two embeddings (bi-encoder, a CLIP dual
     * encoder's image entries included; a SigLIP dual encoder's sigmoid score of it). The results come best first, {@code top_n} of them, each with its
     * document when {@code return_documents} is set: {@code {"text"}} or vLLM's {@code {"multi_modal"}} parts.
     */
    Response rerankOf(JsonView body) {
        String requestId = owner.poolingRequestFields(body);
        String model = owner.service.loras().model(body);
        LoraSelection lora = owner.service.loras().selection(body, model);
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
        ScoringOptions options = ScoringOptions.of(body, owner.service);
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
                scoredScores[i] = owner.service.pooling().biEncoderScore(cosine(queryVector, documentVectors[i]));
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
    static int topN(JsonView body, int count) {
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
    static void refuseMediaTruncation(List<ScoreSides.Entry> one, List<ScoreSides.Entry> two,
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
