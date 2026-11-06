package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The vLLM render endpoints' answers: a generated prompt rendered through the model's tokenizer into the
 * {@code GenerateRequest} a token-in, token-out server runs — one object for chat, messages and responses, and one
 * per prompt for completions. The token ids, the sampling parameters, the per-request {@code parse_context}, the
 * multimodal {@code features} and the per-token character {@code token_offsets} let the generate server reproduce
 * exactly the prompt and settings the prompt-serving endpoint would have used.
 */
final class ScaleOutRender {

    private ScaleOutRender() {
    }

    /**
     * {@code POST /v1/chat/completions/render}: the rendered chat prompt and its sampling parameters. A chat
     * whose messages carry image, audio or video parts renders them through the projector into media rows and
     * answers their hashes, placeholder spans and serialized source in {@code features}; embedding parts travel
     * as serialized rows.
     */
    static Response chat(EngineService service, JsonView body, String requestId) {
        ChatPlan plan = ChatPlan.of(service, body);
        requireRepresentable(plan.n(), plan.bestOf());
        boolean stream = body.optBool("stream", false);
        int[] tokens = plan.media() != null ? ScaleOutFeatures.expandedIds(plan.media()) : plan.tokens();
        int promptRows = plan.media() != null ? plan.media().nTokens() : plan.tokens().length;
        int maxTokens = maxTokens(body, promptRows, service.maxModelLength());
        Map<String, Object> features = plan.media() != null ? ScaleOutFeatures.of(plan.placements()) : null;
        Map<String, Object> answer = answer(requestId, tokens, plan.base(), plan.n(), maxTokens,
                plan.model(), plan.chat(), body, stream, plan.reasoning(), plan.rendered(), features,
                plan.tokenOffsets());
        return Response.ok(ServerJson.write(answer)).build();
    }

    /** {@code POST /v1/completions/render}: one {@code GenerateRequest} per prompt, in prompt order. */
    static Response completions(EngineService service, JsonView body, String baseId) {
        CompletionPlan plan = CompletionPlan.of(service, body);
        requireRepresentable(plan.n(), plan.bestOf());
        boolean stream = body.optBool("stream", false);
        List<Object> answers = new ArrayList<>(plan.prompts().size());
        for (int i = 0; i < plan.prompts().size(); i++) {
            CompletionPlan.Input prompt = plan.prompts().get(i);
            if (prompt.tokens() == null && prompt.media() != null && prompt.placements().isEmpty()) {
                throw ServerJson.badRequest("prompt_embeds is not supported by the render endpoint: an"
                        + " embedding-only prompt has no token ids to render");
            }
            int[] tokens = prompt.tokens() != null ? prompt.tokens() : ScaleOutFeatures.expandedIds(prompt.media());
            int maxTokens = maxTokens(body, prompt.rows(), service.maxModelLength());
            String requestId = plan.prompts().size() == 1 ? baseId : baseId + "-" + i;
            Map<String, Object> features = prompt.placements().isEmpty() ? null
                    : ScaleOutFeatures.of(prompt.placements());
            answers.add(answer(requestId, tokens, plan.base(), plan.n(), maxTokens, plan.model(), null,
                    body, stream, plan.reasoning(), null, features, prompt.offsets()));
        }
        return Response.ok(ServerJson.write(answers)).build();
    }

    /**
     * {@code POST /v1/messages/render}: the Anthropic messages converted through {@link AnthropicRequests} and
     * rendered like a chat prompt, so the tokens match {@code /v1/messages} exactly. Image blocks reach the
     * projector as image parts and travel to the generate server through {@code features}.
     */
    static Response messages(EngineService service, JsonView body, String requestId) {
        String model = service.loras().model(body);
        ChatRequest chat = AnthropicRequests.chat(body, service.options());
        GenerationRequest request = service.loras().apply(body, model,
                AnthropicRequests.request(body, service.logitsProcessors(), service.options().watermark()));
        List<ChatMedia.Item> items = ChatPlan.mediaItems(service, chat);
        String rendered = service.formatChat(chat, !items.isEmpty());
        ChatMedia.Rendered media = items.isEmpty() ? null
                : ChatPlan.renderedMedia(service, chat, rendered, items);
        int[] tokens = media == null ? service.encode(rendered, false, true) : ScaleOutFeatures.expandedIds(
                media.prompt());
        boolean stream = body.optBool("stream", false);
        Map<String, Object> features = media == null ? null : ScaleOutFeatures.of(media.placements());
        List<int[]> offsets = media == null && body.optBool("return_token_offsets", false)
                ? ScaleOutOffsets.of(service, rendered, tokens) : null;
        Map<String, Object> answer = answer(requestId, tokens, request, 1, body.i32("max_tokens"), model,
                chat, body, stream, ReasoningFormats.forChat(service.reasoning(), chat), rendered, features, offsets);
        return Response.ok(ServerJson.write(answer)).build();
    }

    /**
     * {@code POST /v1/responses/render}: the Responses request converted to a chat body ({@link ResponsesRequests})
     * and rendered like {@link #chat}. The stateless render refuses a conversation continued by
     * {@code previous_response_id}, as vLLM's does.
     */
    static Response responses(EngineService service, JsonView body, String requestId) {
        if (body.raw("previous_response_id") != null) {
            throw ServerJson.badRequest("previous_response_id is not supported by the stateless render endpoint");
        }
        if (body.raw("conversation") != null) {
            throw ServerJson.badRequest("'conversation' is not supported by the stateless render endpoint");
        }
        if (body.raw("prompt") != null) {
            throw ServerJson.badRequest("'prompt' (a stored prompt template) is not supported");
        }
        Map<String, Object> chatBody = ResponsesRequests.chatBody(body, List.of());
        if (body.raw("stream") != null) {
            chatBody.put("stream", body.raw("stream"));
        }
        if (body.raw("stream_options") != null) {
            chatBody.put("stream_options", body.raw("stream_options"));
        }
        JsonView chatView = JsonView.of(chatBody, "body");
        ChatPlan plan = ChatPlan.of(service, chatView);
        requireRepresentable(plan.n(), plan.bestOf());
        boolean stream = chatView.optBool("stream", false);
        int[] tokens = plan.media() != null ? ScaleOutFeatures.expandedIds(plan.media()) : plan.tokens();
        int promptRows = plan.media() != null ? plan.media().nTokens() : plan.tokens().length;
        int maxTokens = maxTokens(chatView, promptRows, service.maxModelLength());
        Map<String, Object> features = plan.media() != null ? ScaleOutFeatures.of(plan.placements()) : null;
        Map<String, Object> answer = answer(requestId, tokens, plan.base(), plan.n(), maxTokens,
                plan.model(), plan.chat(), chatView, stream, plan.reasoning(), plan.rendered(), features,
                plan.tokenOffsets());
        return Response.ok(ServerJson.write(answer)).build();
    }

    /** The {@code GenerateRequest} JSON of one rendered prompt. */
    private static Map<String, Object> answer(String requestId, int[] tokens, GenerationRequest request, int n,
                                              int maxTokens, String model, ChatRequest chat, JsonView body,
                                              boolean stream, ReasoningFilter reasoning, String rendered,
                                              Map<String, Object> features, List<int[]> tokenOffsets) {
        Map<String, Object> sampling = ScaleOutRequests.samplingParams(request, body, n, maxTokens, stream);
        Map<String, Object> parseContext = chat == null ? null : ScaleOutRequests.parseContext(body, chat);
        Map<String, Object> answer = ScaleOutRequests.generateRequest(requestId, tokens, sampling, model, stream,
                body, body.optBool("return_token_ids", false), parseContext);
        answer.put("features", features);
        answer.put("token_offsets", ScaleOutOffsets.json(tokenOffsets));
        if (reasoning != null && reasoning.parsesReasoning()) {
            answer.put("reasoning_ended", !reasoning.startsInReasoning(rendered));
            answer.put("reasoning_parser_kwargs", Map.of("chat_template_kwargs",
                    chat != null ? chat.templateVariables() : request.templateVariables()));
        }
        return answer;
    }

    /** Refuses {@code best_of} above {@code n}: vLLM's {@code SamplingParams} carries no candidate count. */
    private static void requireRepresentable(int n, int bestOf) {
        if (bestOf > n) {
            throw ServerJson.badRequest("best_of is not supported by the render endpoints: set best_of to n or"
                    + " drop it");
        }
    }

    /**
     * The token budget the answer carries: the request's {@code max_completion_tokens} or {@code max_tokens}, else
     * the context the prompt leaves (vLLM's {@code get_max_tokens}).
     */
    private static int maxTokens(JsonView body, int promptTokens, int maxModelLength) {
        String key = body.raw("max_completion_tokens") != null ? "max_completion_tokens" : "max_tokens";
        if (body.raw(key) != null) {
            int value = body.i32(key);
            if (value < 0) {
                throw ServerJson.badRequest(key + " must be a non-negative integer, got " + value);
            }
            return value;
        }
        return Math.max(0, maxModelLength - promptTokens);
    }
}
