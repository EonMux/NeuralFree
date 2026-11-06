package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.RepetitionDetection;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The vLLM scale-out wire shapes both directions share: a {@code sampling_params} object as {@code /inference/v1/generate}
 * reads it, and the {@code GenerateRequest} JSON the render endpoints answer with. The sampling fields are the
 * engine's own request vocabulary — the OpenAI fields, vLLM's {@code SamplingParams} names for the same settings and
 * llama.cpp's sampler extensions — so a request the render endpoints emitted feeds the generate endpoint unchanged.
 * Every other field a client sends is refused by name, never dropped.
 */
final class ScaleOutRequests {

    /** The top-level fields {@link #sampling(EngineService, JsonView)} reads; anything else refuses by name. */
    private static final Set<String> SAMPLING_FIELDS = Set.of(
            "n", "max_tokens", "min_tokens",
            "temperature", "top_p", "top_k", "min_p", "typical_p",
            "repetition_penalty", "repeat_penalty", "presence_penalty", "frequency_penalty",
            "repeat_last_n", "min_keep", "seed",
            "stop", "stop_token_ids", "ignore_eos", "include_stop_str_in_output",
            "logprobs", "logprob_token_ids", "prompt_logprobs",
            "skip_special_tokens", "spaces_between_special_tokens",
            "structured_outputs", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
            "structural_tag", "grammar", "json_schema", "response_format",
            "logit_bias", "allowed_token_ids", "bad_words", "repetition_detection",
            "thinking_token_budget", "detokenize", "output_kind", "priority", "cache_salt",
            "dynatemp_range", "dynatemp_exponent", "top_n_sigma", "xtc_probability", "xtc_threshold",
            "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers",
            "mirostat", "mirostat_tau", "mirostat_eta", "adaptive_target", "adaptive_decay", "samplers",
            "watermarking");

    /** The request fields that carry an output constraint ({@link OpenAiRequests#constraint}). */
    private static final String[] CONSTRAINT_FIELDS = {"grammar", "json_schema", "guided_json", "guided_regex",
            "guided_choice", "guided_grammar", "guided_structural_tag", "structural_tag", "response_format",
            "structured_outputs"};

    /** The sampler settings the render endpoints copy from the request body into {@code sampling_params}. */
    private static final String[] SAMPLER_FIELDS = {"temperature", "top_p", "top_k", "min_p", "typical_p",
            "repeat_penalty", "repetition_penalty", "presence_penalty", "frequency_penalty", "repeat_last_n",
            "min_keep", "seed", "dynatemp_range", "dynatemp_exponent", "top_n_sigma", "xtc_probability",
            "xtc_threshold", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
            "dry_sequence_breakers", "mirostat", "mirostat_tau", "mirostat_eta", "adaptive_target", "adaptive_decay",
            "samplers", "watermarking"};

    private ScaleOutRequests() {
    }

    /**
     * The engine request a {@code sampling_params} object asks for: the OpenAI and vLLM sampling fields, the token
     * controls, the cache salt and priority, and llama.cpp's sampler extensions. {@code logprobs} is vLLM's count of
     * reported tokens (the sampled token included), and the constraint fields ride along for the caller to apply.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming an unsupported field or a malformed value
     */
    static GenerationRequest sampling(EngineService service, JsonView sampling) {
        refuseUnsupported(sampling);
        int logprobs = logprobs(sampling);
        if (sampling.raw("logprob_token_ids") != null && logprobs < 0) {
            throw ServerJson.badRequest("sampling_params.logprob_token_ids needs logprobs to be set");
        }
        // vLLM's logprobs is the top-alternatives count; OpenAiRequests maps it to the engine's capture size
        GenerationRequest request;
        try {
            request = OpenAiRequests.sampling(sampling, logprobs, service.vocabSize(), -1,
                    LogprobsMode.RAW_LOGPROBS, service.logitsProcessors(), service.options().watermark());
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest("sampling_params: " + e.getMessage());
        }
        if (sampling.raw("thinking_token_budget") != null) {
            request = request.toBuilder().reasoningBudget(sampling.i32("thinking_token_budget")).build();
        }
        return request;
    }

    /** Rejects every field of {@code sampling} the generate endpoint does not run, naming it. */
    private static void refuseUnsupported(JsonView sampling) {
        for (String key : sampling.keys()) {
            if (!SAMPLING_FIELDS.contains(key)) {
                throw ServerJson.badRequest("sampling_params." + key + " is not supported by this server");
            }
        }
    }

    /** vLLM's {@code logprobs}: a non-negative count of reported tokens per step, or -1 when absent. */
    private static int logprobs(JsonView sampling) {
        Object raw = sampling.raw("logprobs");
        if (raw == null) {
            return -1;
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw ServerJson.badRequest("sampling_params.logprobs must be a non-negative integer or null");
        }
        long value = ((Number) raw).longValue();
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("sampling_params.logprobs must be a non-negative integer, got " + value);
        }
        return (int) value;
    }

    /** Whether {@code sampling} turns detokenization off (vLLM's {@code detokenize}). */
    static boolean detokenize(JsonView sampling) {
        Object raw = sampling.raw("detokenize");
        if (raw == null) {
            return true;
        }
        if (!(raw instanceof Boolean value)) {
            throw ServerJson.badRequest("sampling_params.detokenize must be a boolean");
        }
        return value;
    }

    /**
     * Checks vLLM's {@code output_kind} against the request's {@code stream}: a stream delivers deltas
     * ({@code delta} or {@code cumulative}) and a non-streamed request its final result ({@code final_only}).
     */
    static void requireOutputKind(JsonView sampling, boolean stream) {
        Object raw = sampling.raw("output_kind");
        if (raw == null) {
            return;
        }
        if (!(raw instanceof String kind) || !(kind.equals("delta") || kind.equals("cumulative")
                || kind.equals("final_only"))) {
            throw ServerJson.badRequest("sampling_params.output_kind must be delta, cumulative or final_only");
        }
        if (stream && kind.equals("final_only")) {
            throw ServerJson.badRequest("sampling_params.output_kind final_only cannot be combined with stream true");
        }
        if (!stream && !kind.equals("final_only")) {
            throw ServerJson.badRequest("sampling_params.output_kind must be final_only when stream is false, got '"
                    + kind + "'");
        }
    }

    /**
     * The {@code sampling_params} object a render endpoint answers with, carrying the request's resolved settings:
     * the choices, the token budget, the sampler settings the body set, the token controls and the output constraint.
     * A later {@code /inference/v1/generate} over the returned {@code token_ids} runs exactly this request.
     */
    static Map<String, Object> samplingParams(GenerationRequest request, JsonView body, int n, int maxTokens,
                                              boolean stream) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("n", n);
        out.put("max_tokens", maxTokens);
        if (request.minTokens() > 0) {
            out.put("min_tokens", request.minTokens());
        }
        for (String key : SAMPLER_FIELDS) {
            copy(body, key, out);
        }
        if (!request.stopStrings().isEmpty()) {
            out.put("stop", request.stopStrings());
        }
        int[] stopTokenIds = request.stopTokenIds();
        if (stopTokenIds.length > 0) {
            out.put("stop_token_ids", ints(stopTokenIds));
        }
        if (request.ignoreEos()) {
            out.put("ignore_eos", true);
        }
        if (request.logprobs() > 0) {
            out.put("logprobs", request.logprobs() - 1);
        }
        if (request.namesLogprobTokenIds()) {
            out.put("logprob_token_ids", ints(request.logprobTokenIds()));
        }
        if (request.reasoningBudget() != GenerationRequest.REASONING_BUDGET_DISABLED) {
            out.put("thinking_token_budget", request.reasoningBudget());
        }
        copy(body, "prompt_logprobs", out);
        out.put("skip_special_tokens", request.skipSpecialTokens());
        out.put("spaces_between_special_tokens", request.spacesBetweenSpecialTokens());
        if (request.includeStopString()) {
            out.put("include_stop_str_in_output", true);
        }
        if (!request.badWords().isEmpty()) {
            out.put("bad_words", request.badWords());
        }
        int[] allowed = request.allowedTokenIds();
        if (allowed.length > 0) {
            out.put("allowed_token_ids", ints(allowed));
        }
        if (!request.logitBias().isEmpty()) {
            Map<String, Object> bias = new LinkedHashMap<>();
            for (Map.Entry<Integer, Float> e : request.logitBias().entrySet()) {
                bias.put(String.valueOf(e.getKey()), e.getValue());
            }
            out.put("logit_bias", bias);
        }
        RepetitionDetection repetition = request.repetitionDetection();
        if (repetition.enabled()) {
            Map<String, Object> detection = new LinkedHashMap<>();
            detection.put("max_pattern_size", repetition.maxPatternSize());
            detection.put("min_pattern_size", repetition.minPatternSize());
            detection.put("min_count", repetition.minCount());
            out.put("repetition_detection", detection);
        }
        if (request.priority() != 0) {
            out.put("priority", request.priority());
        }
        if (request.cacheSalt() != null) {
            out.put("cache_salt", request.cacheSalt());
        }
        out.put("detokenize", true);
        out.put("output_kind", stream ? "delta" : "final_only");
        for (String key : CONSTRAINT_FIELDS) {
            copy(body, key, out);
        }
        return out;
    }

    /**
     * One render answer: the rendered token ids, the sampling parameters and the fields a generate request carries
     * beside them ({@code request_id}, {@code model}, the stream switch, {@code return_token_ids} and, for chat, the
     * {@code parse_context} the derender step reads).
     */
    static Map<String, Object> generateRequest(String requestId, int[] tokenIds, Map<String, Object> samplingParams,
                                               String model, boolean stream, JsonView body,
                                               boolean returnTokenIds, Map<String, Object> parseContext) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("request_id", requestId);
        out.put("token_ids", ints(tokenIds));
        out.put("sampling_params", samplingParams);
        out.put("model", model);
        out.put("stream", stream);
        if (stream && body.raw("stream_options") != null) {
            out.put("stream_options", body.raw("stream_options"));
        }
        out.put("return_token_ids", returnTokenIds);
        if (parseContext != null) {
            out.put("parse_context", parseContext);
        }
        return out;
    }

    /**
     * The {@code parse_context} of a chat render answer: the tool definitions rendered back into the OpenAI shape,
     * the tool choice, the parallel-call switch and the reasoning visibility, so a coordinator can send the render
     * answer to a derender step unchanged. {@code history_tool_call_cnt} counts the calls already in the rendered
     * conversation.
     */
    static Map<String, Object> parseContext(JsonView body, com.eonmux.neuralfree.gguf.chat.ChatRequest chat) {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Object> tools = new ArrayList<>();
        for (com.eonmux.neuralfree.gguf.chat.ChatTool tool : chat.tools()) {
            Map<String, Object> entry = tool.templateObject();
            if (tool.strict() != null) {
                @SuppressWarnings("unchecked")
                Map<String, Object> fn = (Map<String, Object>) entry.get("function");
                fn.put("strict", tool.strict());
            }
            tools.add(entry);
        }
        out.put("tools", tools);
        out.put("tool_choice", toolChoice(chat.toolChoice()));
        out.put("parallel_tool_calls", body.raw("parallel_tool_calls") == null
                ? true : body.optBool("parallel_tool_calls", true));
        out.put("include_reasoning", body.optBool("include_reasoning", true));
        Object kwargs = body.raw("chat_template_kwargs");
        out.put("chat_template_kwargs", kwargs instanceof Map<?, ?> ? kwargs : Map.of());
        out.put("history_tool_call_cnt", historyToolCalls(chat));
        return out;
    }

    /** A tool choice in the OpenAI shape ({@code none}, {@code auto}, {@code required} or a named function). */
    private static Object toolChoice(com.eonmux.neuralfree.gguf.chat.ChatRequest.ToolChoice choice) {
        if (choice == null) {
            return "none";
        }
        switch (choice.kind()) {
            case AUTO:
                return "auto";
            case REQUIRED:
                return "required";
            case FUNCTION:
                Map<String, Object> function = new LinkedHashMap<>();
                function.put("name", choice.functionName());
                Map<String, Object> named = new LinkedHashMap<>();
                named.put("type", "function");
                named.put("function", function);
                return named;
            default:
                return "none";
        }
    }

    /** The tool calls already in the rendered messages (assistant {@code tool_calls} entries). */
    private static int historyToolCalls(com.eonmux.neuralfree.gguf.chat.ChatRequest chat) {
        int count = 0;
        for (Map<String, Object> message : chat.messages()) {
            if (message.get("tool_calls") instanceof List<?> calls) {
                count += calls.size();
            }
        }
        return count;
    }

    /** {@code array} as a JSON list of integers. */
    static List<Integer> ints(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int value : values) {
            out.add(value);
        }
        return out;
    }

    /** Copies {@code key} from {@code body} when it is present and not null. */
    private static void copy(JsonView body, String key, Map<String, Object> out) {
        Object raw = body.raw(key);
        if (raw != null) {
            out.put(key, raw);
        }
    }
}
