package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The vLLM derender endpoints: a token-in, token-out generate response turned back into the OpenAI chat-completion
 * or completions response without touching the GPU. The generated tokens are detokenized seeded by the prompt tail,
 * and a chat's content, reasoning and tool calls are read back through the model's reasoning filter and tool parser,
 * so the message matches the coupled chat endpoint's.
 *
 * <p>Non-streamed calls are served here; {@code stream: true} takes the stateless chunk-by-chunk path of
 * {@link ScaleOutDerenderStream}. The request must carry the original {@code chat_request} for tool and reasoning
 * parsing; without one the answer is the raw detokenized text.
 */
final class ScaleOutDerender {

    private ScaleOutDerender() {
    }

    /** One generate response's choice as derender reads it. */
    private record Choice(int[] tokens, String finishReason, Object stopReason, JsonView logprobs) { }

    /**
     * {@code POST /v1/chat/completions/derender}: the {@code generate_response} and the original
     * {@code chat_request} answer a chat completion. The request carries the prompt's token ids (or count) for the
     * usage and to seed detokenization; without a {@code chat_request} the answer is the raw detokenized text.
     */
    static Response chat(EngineService service, JsonView body, String defaultModel) {
        requireNotStreamed(body);
        JsonView generate = body.obj("generate_response");
        List<Choice> choices = choices(service, generate, "generate_response");
        String model = body.optStr("model", defaultModel);
        int promptTokens = body.raw("prompt_tokens") == null ? 0 : body.i32("prompt_tokens");
        int[] promptIds = promptIds(service, body.raw("prompt_token_ids"), generate);
        String renderedPrompt = promptIds == null ? null : service.decode(promptIds);
        JsonView chatRequest = body.raw("chat_request") == null ? null : body.obj("chat_request");
        long completion = 0;
        List<Object> out = new ArrayList<>(choices.size());
        for (int i = 0; i < choices.size(); i++) {
            Choice choice = choices.get(i);
            String text = detokenize(service, choice.tokens(), promptIds);
            Map<String, Object> message = chatRequest == null
                    ? plainMessage(text)
                    : parsedMessage(service, chatRequest, choice.tokens(), renderedPrompt, text);
            Map<String, Object> body_ = new LinkedHashMap<>();
            body_.put("index", i);
            body_.put("message", message);
            body_.put("logprobs", choice.logprobs() == null ? null
                    : chatLogprobs(choice.logprobs(), service));
            String finish = finishReason(choice.finishReason(), message.containsKey("tool_calls"));
            body_.put("finish_reason", finish);
            body_.put("stop_reason", choice.stopReason());
            out.add(body_);
            completion += choice.tokens().length;
        }
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "chatcmpl-" + responseId(generate));
        response.put("object", "chat.completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", model);
        response.put("choices", out);
        response.put("usage", ScaleOutGenerate.usage(promptTokens, completion));
        return Response.ok(ServerJson.write(response)).build();
    }

    /**
     * {@code POST /v1/completions/derender}: one completion choice per choice of the {@code generate_responses}
     * entries, in order, with the {@code prompt_tokens} and {@code prompt_token_ids} entries seeding the usage and
     * the detokenization.
     */
    static Response completions(EngineService service, JsonView body, String defaultModel) {
        requireNotStreamed(body);
        Object raw = body.raw("generate_responses");
        if (!(raw instanceof List<?> responses) || responses.isEmpty()) {
            throw ServerJson.badRequest("generate_responses is required and must be a non-empty array");
        }
        List<Long> promptTokens = numberList(body.raw("prompt_tokens"), responses.size(), "prompt_tokens");
        Object rawPromptIds = body.raw("prompt_token_ids");
        if (rawPromptIds != null && (!(rawPromptIds instanceof List<?> ids) || ids.size() != responses.size())) {
            throw ServerJson.badRequest("prompt_token_ids must be an array parallel to generate_responses");
        }
        String model = body.optStr("model", defaultModel);
        List<Object> out = new ArrayList<>();
        long completion = 0;
        long promptTotal = 0;
        int index = 0;
        for (int i = 0; i < responses.size(); i++) {
            JsonView generate = JsonView.of(responses.get(i), "generate_responses[" + i + "]");
            List<Choice> choices = choices(service, generate, "generate_responses[" + i + "]");
            int[] promptIds = promptIds(service, idsOf(rawPromptIds, i), generate);
            promptTotal += promptTokens.get(i);
            for (Choice choice : choices) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("index", index++);
                entry.put("text", detokenize(service, choice.tokens(), promptIds));
                entry.put("logprobs", choice.logprobs() == null ? null
                        : completionLogprobs(choice.logprobs(), service));
                entry.put("finish_reason", finishReason(choice.finishReason(), false));
                entry.put("stop_reason", choice.stopReason());
                out.add(entry);
                completion += choice.tokens().length;
            }
        }
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "cmpl-" + responseId(JsonView.of(responses.get(0), "generate_responses[0]")));
        response.put("object", "text_completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", model);
        response.put("choices", out);
        response.put("usage", ScaleOutGenerate.usage(promptTotal, completion));
        return Response.ok(ServerJson.write(response)).build();
    }

    // ------------------------------------------------------------------ request plumbing

    /** Streaming derender needs client-held parser state this build cannot replay; refuse it by name. */
    private static void requireNotStreamed(JsonView body) {
        if (body.optBool("stream", false)) {
            throw ServerJson.badRequest("stream must be false or omitted: this server serves non-streamed"
                    + " derendering only");
        }
    }

    /** The generate response's choices, in order; only a tokens-mode response may be derendered. */
    private static List<Choice> choices(EngineService service, JsonView generate, String where) {
        if (generate.raw("output_mode") != null && !"tokens".equals(generate.str("output_mode"))) {
            throw ServerJson.badRequest(where + ".output_mode must be tokens: a detokenized response has already"
                    + " applied its own stop-string truncation");
        }
        Object raw = generate.raw("choices");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(where + ".choices is required and must be a non-empty array");
        }
        List<Choice> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            String at = where + ".choices[" + i + "]";
            JsonView choice = JsonView.of(list.get(i), at);
            Object ids = choice.raw("token_ids");
            if (!(ids instanceof List<?> tokens)) {
                throw ServerJson.badRequest(at + ".token_ids is required: only token-in, token-out responses"
                        + " derender");
            }
            out.add(new Choice(OpenAiRequests.tokens(tokens, service.vocabSize(), at + ".token_ids"),
                    choice.optStr("finish_reason", "stop"), stopReason(choice.raw("stop_reason"), at),
                    choice.raw("logprobs") == null ? null : choice.obj("logprobs")));
        }
        return out;
    }

    /** vLLM's {@code stop_reason}: a string, an integer token id, or null. */
    private static Object stopReason(Object raw, String where) {
        if (raw == null || raw instanceof String || raw instanceof Integer || raw instanceof Long) {
            return raw;
        }
        throw ServerJson.badRequest(where + ".stop_reason must be a string, an integer or null");
    }

    /** The prompt token ids the response carries (at the request level, else the generate response's). */
    private static int[] promptIds(EngineService service, Object raw, JsonView generate) {
        Object ids = raw != null ? raw : generate.raw("prompt_token_ids");
        if (ids == null) {
            return null;
        }
        if (!(ids instanceof List<?> list)) {
            throw ServerJson.badRequest("prompt_token_ids must be an array of token ids");
        }
        return OpenAiRequests.tokens(list, service.vocabSize(), "prompt_token_ids");
    }

    /** The {@code i}-th entry of a nullable list, or null. */
    private static Object idsOf(Object raw, int i) {
        return raw == null ? null : ((List<?>) raw).get(i);
    }

    /** A list of non-negative prompt counts, padded with 0 when absent; each entry must be a non-negative integer. */
    private static List<Long> numberList(Object raw, int size, String key) {
        List<Long> out = new ArrayList<>(size);
        if (raw == null) {
            for (int i = 0; i < size; i++) {
                out.add(0L);
            }
            return out;
        }
        if (!(raw instanceof List<?> list) || list.size() != size) {
            throw ServerJson.badRequest(key + " must be an array parallel to generate_responses");
        }
        for (Object value : list) {
            if (!(value instanceof Number number) || number.longValue() < 0) {
                throw ServerJson.badRequest(key + " must hold non-negative integers");
            }
            out.add(number.longValue());
        }
        return out;
    }

    /** The response id a derender answer carries: its generate response's, else a fresh one. */
    private static String responseId(JsonView generate) {
        Object raw = generate.raw("request_id");
        return raw instanceof String id && !id.isEmpty() ? id : String.valueOf(System.nanoTime());
    }

    // ------------------------------------------------------------------ detokenization

    /**
     * The generated tokens' text, seeded by the last few prompt tokens so the first output token keeps its leading
     * space on a SentencePiece tokenizer (vLLM's derender seeding).
     */
    static String detokenize(EngineService service, int[] tokens, int[] promptIds) {
        if (promptIds == null || promptIds.length == 0) {
            return service.decode(tokens);
        }
        int keep = Math.min(promptIds.length, 8);
        int[] prefix = java.util.Arrays.copyOfRange(promptIds, promptIds.length - keep, promptIds.length);
        int[] all = new int[prefix.length + tokens.length];
        System.arraycopy(prefix, 0, all, 0, prefix.length);
        System.arraycopy(tokens, 0, all, prefix.length, tokens.length);
        String before = service.decode(prefix);
        String together = service.decode(all);
        return together.startsWith(before) ? together.substring(before.length()) : service.decode(tokens);
    }

    // ------------------------------------------------------------------ chat parsing

    /**
     * The message of a chat derender over the model's reasoning filter and the request's tool routing.
     * {@code decoded} is the choice's detokenized text: without reasoning or calls the message content is exactly
     * it, so a tokenizer's piece boundaries (a SentencePiece leading space) do not leak into the answer.
     */
    static Map<String, Object> parsedMessage(EngineService service, JsonView chatBody, int[] tokens,
                                             String renderedPrompt, String decoded) {        ChatRequest chat = OpenAiRequests.chat(chatBody, service.options());
        ReasoningFilter filter = ReasoningFormats.forChat(service.reasoning(), chat);
        ParallelToolCalls parallel = ParallelToolCalls.of(OpenAiRequests.parallelToolCalls(chatBody),
                service.defaultParallelToolCalls(chat));
        ToolRouting routing = service.toolRouting(chat, renderedPrompt, parallel, filter);
        ChatChoice collector = new ChatChoice(routing.newOutput(), chatBody.optBool("include_reasoning", true));
        ReasoningFilter.State state = filter.newState(collectorSink(collector), renderedPrompt);
        for (int token : tokens) {
            state.onToken(token, service.tokenText(token));
        }
        state.finish();
        ToolParse parse = collector.finish();
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        String reasoning = collector.messageReasoning();
        String content = parse.calls().isEmpty() && reasoning.isEmpty() ? decoded : parse.content();
        message.put("content", parse.calls().isEmpty() || !content.isEmpty() ? content : null);
        if (!reasoning.isEmpty()) {
            message.put("reasoning_content", reasoning);
            message.put("reasoning", reasoning);
        }
        if (!parse.calls().isEmpty()) {
            message.put("tool_calls", ChatChoice.toolCallsBody(parse.calls()));
        }
        return message;
    }

    /** Routes the reasoning filter's output into one {@link ChatChoice}. */
    private static ReasoningFilter.Sink collectorSink(ChatChoice collector) {
        return new ReasoningFilter.Sink() {
            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                collector.accept(new GenerationBatch.Text(0, channel, text, null, tokenId, 0));
            }

            @Override
            public void toolCallStarted(String recipient) {
                collector.accept(new GenerationBatch.ToolRecipient(0, recipient));
            }

            @Override
            public void contentFallback() {
                collector.accept(new GenerationBatch.ContentFallback(0));
            }
        };
    }

    /** The raw answer of a derender without a chat request. */
    static Map<String, Object> plainMessage(String text) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", text);
        return message;
    }

    /** {@code tool_calls} when calls were read and the generation stopped for them, as the chat endpoint answers. */
    static String finishReason(String finish, boolean calls) {
        String base = finish == null ? "stop" : finish;
        return calls && base.equals("stop") ? "tool_calls" : base;
    }

    // ------------------------------------------------------------------ logprobs

    /** A generate tokens-shape capture as the chat logprobs object (decoded tokens and bytes). */
    @SuppressWarnings("unchecked")
    static Object chatLogprobs(JsonView logprobs, EngineService service) {
        Object rawContent = logprobs.raw("content");
        if (!(rawContent instanceof List<?> content)) {
            return rawContent;
        }
        List<Object> out = new ArrayList<>(content.size());
        for (Object entry : content) {
            out.add(chatLogprob((Map<String, Object>) entry, service));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("content", out);
        return body;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> chatLogprob(Map<String, Object> entry, EngineService service) {
        if (entry.containsKey("token")) {
            return entry;
        }
        int tokenId = number(entry.get("token_id"), "token_id").intValue();
        String token = service.tokenText(tokenId);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("token", token);
        out.put("logprob", entry.get("logprob"));
        out.put("bytes", bytes(token));
        Object tops = entry.get("top_logprobs");
        if (tops instanceof List<?> list) {
            List<Object> alternatives = new ArrayList<>(list.size());
            for (Object top : list) {
                Map<String, Object> candidate = new LinkedHashMap<>();
                int id = number(((Map<?, ?>) top).get("token_id"), "token_id").intValue();
                String text = service.tokenText(id);
                candidate.put("token", text);
                candidate.put("logprob", ((Map<?, ?>) top).get("logprob"));
                candidate.put("bytes", bytes(text));
                alternatives.add(candidate);
            }
            out.put("top_logprobs", alternatives);
        }
        return out;
    }

    /** A generate tokens-shape capture as the completions legacy logprobs arrays. */
    @SuppressWarnings("unchecked")
    private static Object completionLogprobs(JsonView logprobs, EngineService service) {
        Object rawContent = logprobs.raw("content");
        if (!(rawContent instanceof List<?> content)) {
            return null;
        }
        List<Object> tokens = new ArrayList<>(content.size());
        List<Object> values = new ArrayList<>(content.size());
        List<Object> tops = new ArrayList<>(content.size());
        List<Object> offsets = new ArrayList<>(content.size());
        int at = 0;
        for (Object raw : content) {
            Map<String, Object> entry = (Map<String, Object>) raw;
            int id = number(entry.get("token_id"), "token_id").intValue();
            String text = service.tokenText(id);
            tokens.add(text);
            values.add(entry.get("logprob"));
            Map<String, Object> alternatives = new LinkedHashMap<>();
            if (entry.get("top_logprobs") instanceof List<?> list) {
                for (Object top : list) {
                    int topId = number(((Map<?, ?>) top).get("token_id"), "token_id").intValue();
                    alternatives.putIfAbsent(service.tokenText(topId), ((Map<?, ?>) top).get("logprob"));
                }
            }
            tops.add(alternatives);
            offsets.add(at);
            at += text.length();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", tokens);
        out.put("token_logprobs", values);
        out.put("top_logprobs", tops);
        out.put("text_offset", offsets);
        return out;
    }

    private static List<Object> bytes(String text) {
        List<Object> out = new ArrayList<>();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            out.add(b & 0xff);
        }
        return out;
    }

    private static Number number(Object raw, String key) {
        if (!(raw instanceof Number value)) {
            throw ServerJson.badRequest("logprobs." + key + " must be an integer");
        }
        return value;
    }
}
