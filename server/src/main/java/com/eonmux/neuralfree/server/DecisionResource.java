package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
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

/**
 * The decision endpoints over every decision family the engine evaluates. {@code POST /v1/systemone} is the
 * TypeSafe-compatible API llama.cpp's server serves: a {@code state} and a map of questions (choice, score or noul),
 * every question read through the model's systemone template — the per-option heads (vllm-decision2) one prompt per
 * question, the clef head all questions as one joint prompt, the label families (openjev, lev, nimble, pplx-decider,
 * vllm-d3, lfm2-d1, lfm3-d1) the answer-code logits at the last prompt token, and the modern-bert head (laya) one
 * question-type score at each option's marker — answered per question id with the option probabilities, the
 * confidence and a score legend, over the model's per-type temperatures and score offsets.
 *
 * <p>{@code POST /v1/decisions} is vLLM's Decisions API over the same reads: a text {@code input} and a list of
 * predicate, choice and score questions whose answers echo the request's {@code name}, the raw choice values and the
 * level labels, with vLLM's usage block. The option caps and the {@code joint} refusal are the serving limits of
 * vLLM's decision strategies; a head that reads every question as one prompt (clef) admits {@code joint}, and the
 * heads that read each question on its own refuse it by name.</p>
 *
 * <p>Media follows llama.cpp's decision server: {@code files}/{@code images} data URLs and the image or audio parts
 * of a message state become the prompt's media through the server's projector ({@code --mmproj}), one media marker
 * per item; a family without an image slot, or a server without a projector, answers llama.cpp's 501, and
 * {@code videos} refuse by name.</p>
 */
@Path("/")
public class DecisionResource {

    /** vLLM's refusal for a joint read on a model that reads each question on its own. */
    private static final String JOINT_REFUSAL = "joint: this model reads each question on its own;"
            + " joint reads need DiffusionGemma";
    /** llama.cpp's refusal of a decision request that attaches media without a projector. */
    private static final String NO_PROJECTOR = "This server does not support image input for decisions. For a model"
            + " that supports it, start it with `--mmproj`";
    /** llama.cpp's message for more media than a decision request takes. */
    private static final String TOO_MANY_IMAGES = "too many images, the maximum is " + DecisionModel.MAX_MEDIA;
    /** llama.cpp's message for more audio than a decision request takes. */
    private static final String TOO_MANY_MEDIA = "too many media files, the maximum is " + DecisionModel.MAX_MEDIA;

    @Inject
    EngineService service;

    @Inject
    HttpServerRequest http;

    /** Framework wiring: the service, request and router are injected fields. */
    public DecisionResource() {
    }

    /** Test and manual wiring: builds the resource over an explicit service. */
    public DecisionResource(EngineService service) {
        this(service, (ModelRouter) null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public DecisionResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
        this.http = null;
    }

    private DecisionResource(EngineService service, HttpServerRequest http) {
        this.service = service;
        this.http = http;
    }

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** A routed copy: the target model's service, serving the same HTTP request (its {@code X-Request-Id}). */
    private DecisionResource routed(JsonView body) {
        return new DecisionResource(ModelRouter.target(router, service, body, null), http);
    }

    /** TypeSafe-compatible decisions: a state, a map of questions, one answer per question id. */
    @POST
    @Path("/v1/systemone")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response systemone(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        DecisionResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.systemoneOf(body));
    }

    /** vLLM's Decisions API: one text input, a list of questions, answers with echoed values. */
    @POST
    @Path("/v1/decisions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response decisions(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        DecisionResource target = routed(body);
        ServerJson.requireLoaded(target.service);
        return OpenAiResource.mapped(() -> target.decisionsOf(body));
    }

    // ------------------------------------------------------------------ /v1/systemone

    private Response systemoneOf(JsonView body) {
        DecisionModel model = DecisionModel.of(service);
        List<DecisionQuestions.Question> questions = DecisionQuestions.parseSystemOne(body, model.flags());
        DecisionModel.Media media = systemOneMedia(body, model);
        String modelName = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, modelName);
        DecisionModel.Result evaluated = evaluate(model, questions, media, lora);
        Map<String, Object> answers = new LinkedHashMap<>();
        for (int i = 0; i < questions.size(); i++) {
            answers.put(questions.get(i).id(),
                    DecisionAnswers.systemOne(model.family(), questions.get(i), evaluated.probs().get(i)));
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", evaluated.tokens());
        usage.put("output_tokens", 0);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", modelName);
        out.put("answers", answers);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The systemone body's media, llama.cpp's {@code parse_state}: {@code videos} refuse by name, {@code files} and
     * its alias {@code images} must be data-URL arrays, and a state of chat messages gives up its {@code image_url},
     * {@code input_audio} and (lfm3-d1) {@code video_url} parts. A request with media on a server without a projector,
     * or on a family without an image slot, answers llama.cpp's 501.
     */
    private DecisionModel.Media systemOneMedia(JsonView body, DecisionModel model) {
        Object videos = body.raw("videos");
        if (videos != null && !(videos instanceof List<?> list && list.isEmpty())) {
            throw ServerJson.badRequest("\"videos\" is not supported");
        }
        List<ChatMedia.Item> items = new ArrayList<>();
        for (String key : List.of("files", "images")) {
            Object raw = body.raw(key);
            if (raw == null) {
                continue;
            }
            if (!(raw instanceof List<?> urls)) {
                throw ServerJson.badRequest("\"" + key + "\" must be an array");
            }
            for (Object url : urls) {
                if (!(url instanceof String text) || !text.startsWith("data:")) {
                    throw ServerJson.badRequest("media must be data URLs (data:image/..., data:audio/... or"
                            + " data:video/...;base64,...)");
                }
                if (items.size() >= DecisionModel.MAX_MEDIA) {
                    throw ServerJson.badRequest(TOO_MANY_IMAGES);
                }
                items.add(loadItem(List.of(url), key));
            }
        }
        Object state = stateMedia(body.raw("state"), model, items);
        if (!items.isEmpty() && (!model.family().canUseImages || !service.media().enabled())) {
            throw ServerJson.error(501, NO_PROJECTOR, "not_supported_error");
        }
        // minicpm-ocr-jev reads its fixed two-image question: the state is the OCR text, the images the original
        // and the rendered one (llama.cpp's parse_state)
        if (model.family().fixedQuestion()) {
            if (!(state instanceof String)) {
                throw ServerJson.badRequest("\"state\" must be the OCR text");
            }
            if (items.size() != 2) {
                throw ServerJson.badRequest("this model needs exactly 2 images (the original, then the rendered"
                        + " one), got " + items.size());
            }
        }
        for (ChatMedia.Item item : items) {
            service.media().checkDecisionItem(item);
        }
        return new DecisionModel.Media(state, items);
    }

    /**
     * The state's media parts: a message array, or {@code {"messages": [...]}}, loses its {@code image_url} parts
     * (and the audio and video parts llama.cpp's decision parser takes) to the request's media; every other state is
     * unchanged.
     */
    private Object stateMedia(Object state, DecisionModel model, List<ChatMedia.Item> items) {
        boolean wrapped = state instanceof Map<?, ?> map && map.get("messages") instanceof List<?>;
        Object messages = wrapped ? ((Map<?, ?>) state).get("messages") : state;
        if (!(messages instanceof List<?> list)) {
            return state;
        }
        boolean changed = false;
        List<Object> out = new ArrayList<>(list.size());
        for (Object raw : list) {
            if (!(raw instanceof Map<?, ?> message) || !(message.get("content") instanceof List<?> content)) {
                out.add(raw);
                continue;
            }
            List<Object> kept = new ArrayList<>(content.size());
            boolean messageChanged = false;
            for (Object rawPart : content) {
                if (rawPart instanceof Map<?, ?> part) {
                    String type = part.get("type") instanceof String named ? named : "";
                    if (type.equals("image_url") && part.containsKey("image_url")) {
                        addImage(items, part.get("image_url"), "state.image_url");
                        messageChanged = true;
                        continue;
                    }
                    if (type.equals("input_audio") && part.containsKey("input_audio")) {
                        addAudio(items, part.get("input_audio"), "state.input_audio");
                        messageChanged = true;
                        continue;
                    }
                    if (type.equals("video_url") && part.containsKey("video_url")
                            && model.family().type.startsWith("lfm3-d1")) {
                        addImage(items, part.get("video_url"), "state.video_url");
                        messageChanged = true;
                        continue;
                    }
                }
                kept.add(rawPart);
            }
            if (messageChanged) {
                Map<String, Object> rewritten = new LinkedHashMap<>();
                for (Map.Entry<?, ?> entry : message.entrySet()) {
                    rewritten.put(String.valueOf(entry.getKey()),
                            "content".equals(String.valueOf(entry.getKey())) ? kept : entry.getValue());
                }
                out.add(rewritten);
                changed = true;
            } else {
                out.add(raw);
            }
        }
        if (!changed) {
            return state;
        }
        if (wrapped) {
            Map<String, Object> rewritten = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) state).entrySet()) {
                rewritten.put(String.valueOf(entry.getKey()),
                        "messages".equals(String.valueOf(entry.getKey())) ? out : entry.getValue());
            }
            return rewritten;
        }
        return out;
    }

    /** Adds one image reference (a URL string or an object with {@code url}) to the request's media. */
    private void addImage(List<ChatMedia.Item> items, Object reference, String field) {
        Object value = reference instanceof Map<?, ?> map ? map.get("url") : reference;
        if (!(value instanceof String url) || !url.startsWith("data:")) {
            throw ServerJson.badRequest("media must be data URLs (data:image/..., data:audio/... or"
                    + " data:video/...;base64,...)");
        }
        if (items.size() >= DecisionModel.MAX_MEDIA) {
            throw ServerJson.badRequest(TOO_MANY_IMAGES);
        }
        items.add(loadItem(List.of(url), field));
    }

    /** Adds one {@code input_audio} part (its {@code data} or {@code url}) to the request's media. */
    private void addAudio(List<ChatMedia.Item> items, Object part, String field) {
        Object value = part instanceof Map<?, ?> map
                ? (map.get("data") != null ? map.get("data") : map.get("url")) : part;
        if (!(value instanceof String data)) {
            throw ServerJson.badRequest("audio must be base64 data");
        }
        if (items.size() >= DecisionModel.MAX_MEDIA) {
            throw ServerJson.badRequest(TOO_MANY_MEDIA);
        }
        items.add(audioItem(data, field));
    }

    /** One data-URL item, mapped to a 400 naming the field when it does not load. */
    private ChatMedia.Item loadItem(List<?> urls, String field) {
        try {
            List<ChatMedia.Item> loaded = service.media().decisionItems(urls, field);
            return loaded.get(0);
        } catch (MediaInputException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** One audio item, mapped to a 400 naming the field when it does not decode. */
    private ChatMedia.Item audioItem(String data, String field) {
        try {
            return service.media().decisionAudio(data, field);
        } catch (MediaInputException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    // ------------------------------------------------------------------ /v1/decisions

    private Response decisionsOf(JsonView body) {
        DecisionModel model = DecisionModel.of(service);
        OpenAiRequests.cacheSalt(body);
        safetyIdentifier(body);
        List<DecisionQuestions.Question> questions = DecisionQuestions.parseVllm(body, model.flags(),
                model.vllmMaxOptions());
        String text = inputText(body);
        Object joint = body.raw("joint");
        if (joint != null && !(joint instanceof Boolean)) {
            throw ServerJson.badRequest("\"joint\" must be a boolean");
        }
        if (Boolean.TRUE.equals(joint) && !model.family().jointReads) {
            throw ServerJson.badRequest(JOINT_REFUSAL);
        }
        String modelName = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, modelName);
        DecisionModel.Result evaluated = evaluate(model, questions, DecisionModel.Media.none(text), lora);
        List<Map<String, Object>> answers = new ArrayList<>();
        for (int i = 0; i < questions.size(); i++) {
            answers.add(DecisionAnswers.vllm(questions.get(i),
                    questions.get(i).requestOrder(evaluated.probs().get(i))));
        }
        Map<String, Object> inputDetails = new LinkedHashMap<>();
        inputDetails.put("cached_tokens", 0);
        inputDetails.put("cache_write_tokens", 0);
        Map<String, Object> outputDetails = new LinkedHashMap<>();
        outputDetails.put("reasoning_tokens", 0);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", evaluated.tokens());
        usage.put("input_tokens_details", inputDetails);
        usage.put("output_tokens", 0);
        usage.put("output_tokens_details", outputDetails);
        usage.put("total_tokens", evaluated.tokens());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", modelName);
        out.put("answers", answers);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The request's {@code safety_identifier}: optional metadata, a string of at most 128 characters. */
    private void safetyIdentifier(JsonView body) {
        Object safety = body.raw("safety_identifier");
        if (safety != null && (!(safety instanceof String named) || named.length() > 128)) {
            throw ServerJson.badRequest("\"safety_identifier\" must be a string of at most 128 characters");
        }
    }

    /**
     * The request's input text: a string, or chat messages (a string content, or {@code input_text} parts joined
     * with a newline) joined with a blank line, as vLLM's {@code input_text} joins them; image parts refuse, as
     * vLLM's Decisions validator does.
     */
    private String inputText(JsonView body) {
        Object raw = body.raw("input");
        if (raw instanceof String text) {
            if (text.length() > 10_485_760) {
                throw ServerJson.badRequest("\"input\" must be at most 10485760 characters");
            }
            return text;
        }
        if (!(raw instanceof List<?> messages)) {
            throw ServerJson.badRequest("\"input\" must be text or an array of messages");
        }
        if (messages.size() > 131_072) {
            throw ServerJson.badRequest("\"input\" must hold at most 131072 messages");
        }
        List<String> texts = new ArrayList<>();
        for (Object rawMessage : messages) {
            texts.add(messageText(rawMessage));
        }
        return String.join("\n\n", texts);
    }

    /** One input message's text: its string content, or its {@code input_text} parts joined with a newline. */
    private String messageText(Object rawMessage) {
        if (!(rawMessage instanceof Map<?, ?> message)) {
            throw ServerJson.badRequest("input: every message must be an object");
        }
        Object type = message.get("type");
        if (type != null && !"message".equals(type)) {
            throw ServerJson.badRequest("input: message \"type\" must be \"message\"");
        }
        if (!"user".equals(message.get("role"))) {
            throw ServerJson.badRequest("input: message \"role\" must be \"user\"");
        }
        Object content = message.get("content");
        if (content instanceof String text) {
            return text;
        }
        if (!(content instanceof List<?> parts)) {
            throw ServerJson.badRequest("input: message \"content\" must be text or an array of input_text parts");
        }
        List<String> texts = new ArrayList<>();
        for (Object rawPart : parts) {
            if (rawPart instanceof Map<?, ?> part && "input_image".equals(part.get("type"))) {
                throw ServerJson.badRequest("Decisions supports text input only; images are not supported");
            }
            if (!(rawPart instanceof Map<?, ?> part) || !"input_text".equals(part.get("type"))
                    || !(part.get("text") instanceof String text)) {
                throw ServerJson.badRequest("input: every content part must be an object with \"type\":"
                        + " \"input_text\" and text");
            }
            texts.add(text);
        }
        return String.join("\n", texts);
    }

    // ------------------------------------------------------------------ evaluation

    /** The model's read of every question, with the forward failures mapped to the server's error shapes. */
    private DecisionModel.Result evaluate(DecisionModel model, List<DecisionQuestions.Question> questions,
                                          DecisionModel.Media media, LoraSelection lora) {
        try {
            return model.read(questions, media.state(), media.items(), lora);
        } catch (IOException e) {
            throw ServerJson.error(500, "the decision forward failed: " + e.getMessage(), "server_error");
        } catch (MediaInputException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        } catch (IllegalStateException e) {
            throw ServerJson.error(500, String.valueOf(e.getMessage()), "server_error");
        }
    }
}
