package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The stateless streaming derender (vLLM's {@code stream: true} on the derender endpoints): one call per chunk of a
 * generate stream, carrying the chunks seen so far in a client-held {@code stream_state} and answering the
 * derendered OpenAI chunk plus the updated state. No session lives on the server; the state carries the output
 * tokens, their chunk lengths, the sent role switch and the stable tool-call ids.
 *
 * <p>Each call replays the accumulated tokens through the model's reasoning filter and tool parser and derives the
 * delta from what the previous token prefix parsed to, so parser state never has to be serialized. This is O(n) per
 * chunk; vLLM's own streaming derender pays the same replay cost.</p>
 */
final class ScaleOutDerenderStream {

    /** The {@code stream_state} members this server reads and echoes; anything else is refused by name. */
    private static final Set<String> STATE_FIELDS = Set.of("prev_tokens", "prefix_offset", "read_offset",
            "role_sent", "logprob_context_token_ids", "logprob_text_offset", "output_token_ids",
            "output_chunk_lens", "tools_streamed", "last_tool_call_ids");

    private ScaleOutDerenderStream() {
    }

    /** One generate chunk's choice: its generated tokens, finish reason (or null) and logprobs (or null). */
    private record Chunk(int[] tokens, String finishReason, JsonView logprobs) { }

    /** The client-carried state, parsed and validated. */
    private record State(Map<String, Object> echo, List<Integer> tokens, List<Integer> lengths, boolean roleSent,
                         boolean toolsStreamed, List<String> toolCallIds, int logprobOffset) {

        int[] tokenIds() {
            int[] out = new int[tokens.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = tokens.get(i);
            }
            return out;
        }

        /** A copy carrying {@code chunk} more tokens and the tool-call ids of the answer. */
        State advanced(int[] chunk, List<String> ids, boolean tools, int logprobOffset) {
            List<Integer> tokens = new ArrayList<>(this.tokens);
            List<Integer> lengths = new ArrayList<>(this.lengths);
            for (int token : chunk) {
                tokens.add(token);
            }
            if (chunk.length > 0) {
                lengths.add(chunk.length);
            }
            return new State(echo, tokens, lengths, true, tools || !ids.isEmpty(), ids, logprobOffset);
        }

        Map<String, Object> json() {
            Map<String, Object> out = new LinkedHashMap<>(echo);
            out.put("role_sent", roleSent);
            out.put("tools_streamed", toolsStreamed);
            out.put("output_token_ids", tokens);
            out.put("output_chunk_lens", lengths);
            out.put("last_tool_call_ids", toolCallIds);
            out.put("logprob_text_offset", logprobOffset);
            return out;
        }
    }

    /** The parsed message of a token prefix: content, reasoning and calls. */
    private record Parsed(String content, String reasoning, List<ToolCall> calls) { }

    /**
     * One streaming chat derender call: the OpenAI chat chunk of this generate chunk and the state to carry to the
     * next call.
     */
    static Response chat(EngineService service, JsonView body, String defaultModel) {
        JsonView chunkView = body.obj("generate_chunk");
        Chunk chunk = chunk(service, chunkView, "generate_chunk");
        State state = state(body, service);
        int[] previous = state.tokenIds();
        int[] all = concat(previous, chunk.tokens());
        int[] promptIds = promptIds(service, body, chunkView);
        String renderedPrompt = promptIds == null ? null : service.decode(promptIds);
        JsonView chatRequest = body.raw("chat_request") == null ? null : body.obj("chat_request");
        String beforeText = ScaleOutDerender.detokenize(service, previous, promptIds);
        String afterText = ScaleOutDerender.detokenize(service, all, promptIds);
        Parsed before = chatRequest == null ? new Parsed(beforeText, "", List.of())
                : parsed(service, chatRequest, previous, renderedPrompt, beforeText);
        Parsed after = chatRequest == null ? new Parsed(afterText, "", List.of())
                : parsed(service, chatRequest, all, renderedPrompt, afterText);
        List<String> ids = callIds(state.toolCallIds(), after.calls());
        String model = body.optStr("model", defaultModel);
        Map<String, Object> delta = new LinkedHashMap<>();
        if (!state.roleSent()) {
            delta.put("role", "assistant");
        }
        String content = appended(before.content(), after.content());
        if (!content.isEmpty()) {
            delta.put("content", content);
        }
        String reasoning = appended(before.reasoning(), after.reasoning());
        if (!reasoning.isEmpty()) {
            delta.put("reasoning", reasoning);
            delta.put("reasoning_content", reasoning);
        }
        List<Object> toolDeltas = toolDeltas(before.calls(), after.calls(), ids);
        if (!toolDeltas.isEmpty()) {
            delta.put("tool_calls", toolDeltas);
        }
        String finish = finish(chunk.finishReason(), !after.calls().isEmpty());
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        if (chunk.logprobs() != null) {
            choice.put("logprobs", ScaleOutDerender.chatLogprobs(chunk.logprobs(), service));
        }
        choice.put("finish_reason", finish);
        Map<String, Object> answer = chunkHead("chat.completion.chunk", "chatcmpl-", chunkView, model);
        answer.put("choices", List.of(choice));
        State advanced = state.advanced(chunk.tokens(), ids, !after.calls().isEmpty(),
                state.logprobOffset());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("chunk", answer);
        out.put("stream_state", advanced.json());
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * One streaming completions derender call: the OpenAI completions chunk of this generate chunk and the state to
     * carry to the next call.
     */
    static Response completions(EngineService service, JsonView body, String defaultModel) {
        JsonView chunkView = body.obj("generate_chunk");
        Chunk chunk = chunk(service, chunkView, "generate_chunk");
        State state = state(body, service);
        int[] previous = state.tokenIds();
        int[] all = concat(previous, chunk.tokens());
        int[] promptIds = promptIds(service, body, chunkView);
        String before = ScaleOutDerender.detokenize(service, previous, promptIds);
        String after = ScaleOutDerender.detokenize(service, all, promptIds);
        String text = appended(before, after);
        String model = body.optStr("model", defaultModel);
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("text", text);
        int offset = state.logprobOffset();
        choice.put("logprobs", chunk.logprobs() == null ? null
                : completionLogprobs(chunk.logprobs(), service, offset));
        choice.put("finish_reason", chunk.finishReason());
        Map<String, Object> answer = chunkHead("text_completion", "cmpl-", chunkView, model);
        answer.put("choices", List.of(choice));
        int advancedOffset = offset + logprobTextLength(chunk.logprobs(), service);
        State advanced = state.advanced(chunk.tokens(), state.toolCallIds(), state.toolsStreamed(),
                advancedOffset);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("chunk", answer);
        out.put("stream_state", advanced.json());
        return Response.ok(ServerJson.write(out)).build();
    }

    // ------------------------------------------------------------------ request plumbing

    /** The chunk head: id, object, created and model. */
    private static Map<String, Object> chunkHead(String object, String idPrefix, JsonView chunk, String model) {
        Map<String, Object> head = new LinkedHashMap<>();
        Object named = chunk.raw("request_id");
        head.put("id", idPrefix + (named instanceof String id && !id.isEmpty() ? id
                : String.valueOf(System.nanoTime())));
        head.put("object", object);
        head.put("created", System.currentTimeMillis() / 1000);
        head.put("model", model);
        return head;
    }

    /** The one choice of a generate chunk. */
    private static Chunk chunk(EngineService service, JsonView generate, String where) {
        if (generate.raw("output_mode") != null && !"tokens".equals(generate.str("output_mode"))) {
            throw ServerJson.badRequest(where + ".output_mode must be tokens: a detokenized chunk has already"
                    + " applied its own stop-string truncation");
        }
        Object raw = generate.raw("choices");
        if (!(raw instanceof List<?> choices) || choices.size() != 1) {
            throw ServerJson.badRequest(where + ".choices must hold exactly one choice: streaming derender tracks"
                    + " one sequence per stream_state");
        }
        JsonView choice = JsonView.of(choices.get(0), where + ".choices[0]");
        Object ids = choice.raw("token_ids");
        if (!(ids instanceof List<?> list)) {
            throw ServerJson.badRequest(where + ".choices[0].token_ids is required: only token-in, token-out"
                    + " chunks derender");
        }
        return new Chunk(OpenAiRequests.tokens(list, service.vocabSize(), where + ".choices[0].token_ids"),
                choice.optStr("finish_reason", null),
                choice.raw("logprobs") == null ? null : choice.obj("logprobs"));
    }

    /** The prompt token ids the call carries (at the request level, else the chunk's). */
    private static int[] promptIds(EngineService service, JsonView body, JsonView chunk) {
        Object ids = body.raw("prompt_token_ids") != null ? body.raw("prompt_token_ids")
                : chunk.raw("prompt_token_ids");
        if (ids == null) {
            return null;
        }
        if (!(ids instanceof List<?> list)) {
            throw ServerJson.badRequest("prompt_token_ids must be an array of token ids");
        }
        return OpenAiRequests.tokens(list, service.vocabSize(), "prompt_token_ids");
    }

    /** The client-carried state, or empty state on the first chunk. */
    private static State state(JsonView body, EngineService service) {
        if (body.raw("stream_state") == null) {
            return new State(defaults(), List.of(), List.of(), false, false, List.of(), 0);
        }
        JsonView raw = body.obj("stream_state");
        for (String key : raw.keys()) {
            if (!STATE_FIELDS.contains(key)) {
                throw ServerJson.badRequest("stream_state." + key + " is not supported");
            }
        }
        List<Integer> tokens = ints(raw.raw("output_token_ids"), "stream_state.output_token_ids", 0);
        List<Integer> lengths = ints(raw.raw("output_chunk_lens"), "stream_state.output_chunk_lens", 1);
        long sum = 0;
        for (int length : lengths) {
            sum += length;
        }
        if (sum != tokens.size()) {
            throw ServerJson.badRequest("stream_state.output_chunk_lens must sum to output_token_ids length "
                    + tokens.size() + ", got " + sum);
        }
        int[] tokenArray = new int[tokens.size()];
        for (int i = 0; i < tokenArray.length; i++) {
            tokenArray[i] = tokens.get(i);
        }
        OpenAiRequests.tokens(intList(tokenArray), service.vocabSize(), "stream_state.output_token_ids");
        Map<String, Object> echo = defaults();
        for (String key : new String[]{"prev_tokens", "prefix_offset", "read_offset",
                "logprob_context_token_ids"}) {
            if (raw.raw(key) != null) {
                echo.put(key, raw.raw(key));
            }
        }
        int offset = raw.raw("logprob_text_offset") == null ? 0 : raw.i32("logprob_text_offset");
        if (offset < 0) {
            throw ServerJson.badRequest("stream_state.logprob_text_offset must be non-negative");
        }
        return new State(echo, tokens, lengths, raw.optBool("role_sent", false),
                raw.optBool("tools_streamed", false), strings(raw.raw("last_tool_call_ids")), offset);
    }

    /** The state fields with their defaults, so a response always carries the full shape. */
    private static Map<String, Object> defaults() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("prev_tokens", List.of());
        out.put("prefix_offset", 0);
        out.put("read_offset", 0);
        out.put("logprob_context_token_ids", List.of());
        return out;
    }

    /** A nullable JSON array of integers, each at least {@code min}. */
    private static List<Integer> ints(Object raw, String key, int min) {
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest(key + " must be an array of integers");
        }
        List<Integer> out = new ArrayList<>(list.size());
        for (Object value : list) {
            if (!(value instanceof Number number) || number.longValue() < min
                    || number.longValue() > Integer.MAX_VALUE) {
                throw ServerJson.badRequest(key + " must hold integers of at least " + min);
            }
            out.add(number.intValue());
        }
        return out;
    }

    /** A nullable JSON array of strings. */
    private static List<String> strings(Object raw) {
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("stream_state.last_tool_call_ids must be an array of strings");
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object value : list) {
            if (!(value instanceof String text)) {
                throw ServerJson.badRequest("stream_state.last_tool_call_ids must be an array of strings");
            }
            out.add(text);
        }
        return out;
    }

    private static List<Integer> intList(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int value : values) {
            out.add(value);
        }
        return out;
    }

    // ------------------------------------------------------------------ parsing

    /**
     * The message parts the token prefix parses to under the chat request's filter and tool routing.
     * {@code decoded} is the prefix's detokenized text, the content when the filter splits nothing off.
     */
    private static Parsed parsed(EngineService service, JsonView chatBody, int[] tokens, String renderedPrompt,
                                 String decoded) {
        ChatRequest chat = OpenAiRequests.chat(chatBody, service.options());
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
        String content = parse.calls().isEmpty() && collector.messageReasoning().isEmpty()
                ? decoded : parse.content();
        return new Parsed(content, collector.messageReasoning(), parse.calls());
    }

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

    // ------------------------------------------------------------------ deltas

    /** The text {@code after} adds past {@code before}; empty when it does not extend it. */
    private static String appended(String before, String after) {
        return after.length() > before.length() && after.startsWith(before)
                ? after.substring(before.length()) : "";
    }

    /** Stable call ids: the state's id per index, and the parser's fresh id for a new call. */
    private static List<String> callIds(List<String> previous, List<ToolCall> calls) {
        List<String> out = new ArrayList<>(calls.size());
        for (int i = 0; i < calls.size(); i++) {
            out.add(i < previous.size() ? previous.get(i) : calls.get(i).id());
        }
        return out;
    }

    /** The OpenAI {@code tool_calls} deltas: a new call's start, then the arguments it adds. */
    private static List<Object> toolDeltas(List<ToolCall> before, List<ToolCall> after, List<String> ids) {
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < after.size(); i++) {
            ToolCall call = after.get(i);
            if (i >= before.size()) {
                Map<String, Object> function = new LinkedHashMap<>();
                function.put("name", call.name());
                function.put("arguments", call.arguments());
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("index", i);
                entry.put("id", ids.get(i));
                entry.put("type", "function");
                entry.put("function", function);
                out.add(entry);
            } else {
                String delta = appended(before.get(i).arguments(), call.arguments());
                if (!delta.isEmpty()) {
                    out.add(Map.of("index", i, "function", Map.of("arguments", delta)));
                }
            }
        }
        return out;
    }

    /** The finish reason, naming {@code tool_calls} when calls were parsed and generation stopped normally. */
    private static String finish(String finish, boolean calls) {
        return calls && "stop".equals(finish) ? "tool_calls" : finish;
    }

    /** The tokens of {@code previous} followed by {@code next}. */
    private static int[] concat(int[] previous, int[] next) {
        int[] out = new int[previous.length + next.length];
        System.arraycopy(previous, 0, out, 0, previous.length);
        System.arraycopy(next, 0, out, previous.length, next.length);
        return out;
    }

    // ------------------------------------------------------------------ logprobs

    /** A tokens-shape capture as the completions legacy arrays, its offsets continuing at {@code offset}. */
    private static Object completionLogprobs(JsonView logprobs, EngineService service, int offset) {
        Object rawContent = logprobs.raw("content");
        if (!(rawContent instanceof List<?> content)) {
            return null;
        }
        List<Object> tokens = new ArrayList<>(content.size());
        List<Object> values = new ArrayList<>(content.size());
        List<Object> tops = new ArrayList<>(content.size());
        List<Object> offsets = new ArrayList<>(content.size());
        int at = offset;
        for (Object raw : content) {
            Map<?, ?> entry = (Map<?, ?>) raw;
            int id = tokenId(entry);
            String text = service.tokenText(id);
            tokens.add(text);
            values.add(entry.get("logprob"));
            Map<String, Object> alternatives = new LinkedHashMap<>();
            if (entry.get("top_logprobs") instanceof List<?> list) {
                for (Object top : list) {
                    int topId = tokenId((Map<?, ?>) top);
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

    /** The character length the capture's token texts advance the streaming completions offset. */
    private static int logprobTextLength(JsonView logprobs, EngineService service) {
        Object rawContent = logprobs == null ? null : logprobs.raw("content");
        if (!(rawContent instanceof List<?> content)) {
            return 0;
        }
        int length = 0;
        for (Object raw : content) {
            length += service.tokenText(tokenId((Map<?, ?>) raw)).length();
        }
        return length;
    }

    /** One logprobs entry's integer token id. */
    private static int tokenId(Map<?, ?> entry) {
        Object raw = entry.get("token_id");
        if (!(raw instanceof Number number)) {
            throw ServerJson.badRequest("logprobs entries must carry integer token_id values");
        }
        return number.intValue();
    }
}
