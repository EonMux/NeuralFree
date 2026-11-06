package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.grpc.CandidateTokens;
import com.eonmux.neuralfree.grpc.DecodingParameters;
import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.RandomSampling;
import com.eonmux.neuralfree.grpc.ResponseOptions;
import com.eonmux.neuralfree.grpc.StoppingCriteria;
import com.eonmux.neuralfree.grpc.TokenIds;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * One gRPC {@link GenerateRequest} made ready for the engine: the resolved served model name, its prompt tokens,
 * the sampling request, the output constraint, the LoRA selection and the response flags. It is the gRPC peer of
 * {@link CompletionPlan}; both build the same {@link GenerationBatch.Job}s, so the gRPC surface honours the sampling,
 * stopping, constraint and logprobs settings the HTTP surfaces honour.
 *
 * <p>A field the engine cannot honour is refused by name ({@code INVALID_ARGUMENT} or {@code UNIMPLEMENTED}), never
 * silently ignored: the candidate selections the engine does not support, a stop string under {@code output_text}
 * false, a seed below the random-seed sentinel, a token id outside the vocabulary, and a LoRA adapter that is not
 * loaded. Immutable.</p>
 *
 * @param requestId       the client's request id, empty when it named none
 * @param model           the served model name responses carry
 * @param prompt          the prompt token ids, after truncation
 * @param base            the sampling request of sequence 0
 * @param constraint      the output constraint, or null
 * @param reasoning       the reasoning filter the engine routes output with
 * @param sequences       the sequences per prompt (`num_sequences`, at least 1)
 * @param outputText      whether the response carries text
 * @param outputTokenIds  whether the response carries the generated token ids
 * @param outputLogprobs  whether the response carries the sampled token's log probability
 * @param outputCandidates whether the response carries the alternatives after the sampled token
 * @param promptTokenIds  whether the prompt info carries the prompt token ids
 */
record GrpcRequestPlan(String requestId, String model, int[] prompt, GenerationRequest base,
                       ConstraintSpec constraint, ReasoningFilter reasoning, int sequences,
                       boolean outputText, boolean outputTokenIds, boolean outputLogprobs,
                       boolean outputCandidates, boolean promptTokenIds) {

    GrpcRequestPlan {
        prompt = prompt.clone();
    }

    /** Plans {@code request} over the model {@code service} serves. */
    static GrpcRequestPlan of(EngineService service, GenerateRequest request) {
        service.requireLoaded();
        int vocab = service.vocabSize();
        int[] prompt = prompt(service, request, vocab);
        ResponseOptions options = request.getResponse();
        StoppingCriteria stopping = request.getStopping();
        boolean outputText = !options.hasOutputText() || options.getOutputText();
        if (!outputText && stopping.getStopStringsCount() > 0) {
            throw GrpcFailures.invalid("stopping.stop_strings must be empty when response.output_text is false"
                    + " (the stop strings cannot be detected without detokenizing)");
        }
        Logprobs logprobs = Logprobs.of(options, vocab, prompt.length);
        GenerationRequest base = request(service, request, vocab, options, logprobs);
        OpenAiRequests.requireServedPriority(base, service.schedulingPolicy());
        String model = service.served().resolve(JsonView.of(Map.of("model", request.getModel()), "request"));
        return new GrpcRequestPlan(request.getRequestId(), model, prompt, base,
                constraint(request.getDecoding()), service.reasoning(), sequences(request),
                outputText, options.getOutputTokenIds(), options.getOutputLogprobs(),
                logprobs.outputCandidates, options.getPromptTokenIds());
    }

    /** The sequences per prompt the request asks for ({@code num_sequences}, at least 1). */
    private static int sequences(GenerateRequest request) {
        int sequences = request.getSampling().getNumSequences() == 0
                ? 1 : request.getSampling().getNumSequences();
        if (sequences < 1 || sequences > OpenAiRequests.MAX_CHOICES) {
            throw GrpcFailures.invalid("sampling.num_sequences must be in [1, " + OpenAiRequests.MAX_CHOICES
                    + "], got " + sequences);
        }
        OpenAiRequests.requireGenerationBudget(1, sequences);
        return sequences;
    }

    /** The engine request: the sampler, stopping rules, decoding parameters, KV settings and LoRA selection. */
    private static GenerationRequest request(EngineService service, GenerateRequest request, int vocab,
                                             ResponseOptions options, Logprobs logprobs) {
        GenerationRequest.Builder builder = GenerationRequest.builder()
                .skipSpecialTokens(!options.hasSkipSpecialTokens() || options.getSkipSpecialTokens());
        sampler(builder, request, request.getSampling());
        stopping(builder, request.getStopping(), vocab);
        decoding(builder, request, vocab);
        if (request.hasKv()) {
            String salt = salt(request.getKv().getCacheSalt());
            if (request.getKv().getBypassPrefixCache()) {
                salt = UUID.randomUUID().toString();
            }
            builder.cacheSalt(salt);
        }
        logprobs(logprobs, builder);
        builder.priority(request.getPriority());
        String lora = request.getLoraName();
        if (!lora.isEmpty()) {
            requireAdapter(service, lora);
            builder.lora(LoraSelection.of(lora));
        }
        try {
            return builder.build();
        } catch (IllegalArgumentException e) {
            throw GrpcFailures.invalid(e.getMessage());
        }
    }

    /** Applies the request's logprob selections to {@code builder}. */
    private static void logprobs(Logprobs logprobs, GenerationRequest.Builder builder) {
        if (logprobs.outputEntries > 0) {
            builder.logprobs(logprobs.outputEntries);
        }
        if (logprobs.outputTokenIds != null) {
            builder.logprobTokenIds(logprobs.outputTokenIds);
        }
        if (logprobs.promptEntries != 0) {
            builder.promptLogprobs(logprobs.promptEntries);
        }
        if (logprobs.promptTokenIds != null) {
            builder.promptLogprobTokenIds(logprobs.promptTokenIds);
        }
    }

    /** The engine jobs of the sequences, sequence {@code i} sampling with the request's seed offset. */
    List<GenerationBatch.Job> jobs() {
        List<GenerationBatch.Job> jobs = new ArrayList<>(sequences);
        for (int i = 0; i < sequences; i++) {
            jobs.add(new GenerationBatch.Job(prompt, OpenAiRequests.forChoice(base, i), null, constraint, reasoning));
        }
        return jobs;
    }

    /** A copy of the prompt tokens. */
    @Override
    public int[] prompt() {
        return prompt.clone();
    }

    // ------------------------------------------------------------------ prompt

    private static int[] prompt(EngineService service, GenerateRequest request, int vocab) {
        int[] tokens;
        switch (request.getPromptCase()) {
            case TEXT:
                tokens = service.encode(request.getText(), true, true);
                if (tokens.length == 0) {
                    throw GrpcFailures.invalid("prompt.text: the text encodes to no tokens");
                }
                break;
            case TOKEN_IDS:
                tokens = ids(request.getTokenIds(), vocab, "prompt.token_ids");
                if (tokens.length == 0) {
                    throw GrpcFailures.invalid("prompt.token_ids must not be empty");
                }
                break;
            default:
                throw GrpcFailures.invalid("the request names no prompt: set prompt.text or prompt.token_ids");
        }
        long keep = Integer.toUnsignedLong(request.getTruncatePromptTokens());
        if (keep == 0 || keep >= tokens.length) {
            return tokens;
        }
        int n = (int) keep;
        int[] cut = new int[n];
        System.arraycopy(tokens, 0, cut, 0, n);
        return cut;
    }

    /** The ids of a {@link TokenIds} message, each inside {@code [0, vocab)}. */
    static int[] ids(TokenIds ids, int vocab, String field) {
        List<Integer> values = ids.getIdsList();
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            long id = Integer.toUnsignedLong(values.get(i));
            if (id >= vocab) {
                throw GrpcFailures.invalid(field + ": token id " + id + " is outside the vocabulary [0, " + vocab
                        + ")");
            }
            out[i] = (int) id;
        }
        return out;
    }

    // ------------------------------------------------------------------ sampling and stopping

    private static void sampler(GenerationRequest.Builder builder, GenerateRequest request, RandomSampling sampling) {
        SamplerConfig.Builder sampler = GenerationRequest.defaults().sampler().toBuilder();
        if (request.hasTemperature()) {
            sampler.temperature(finite("temperature", request.getTemperature()));
        }
        if (sampling.hasTopK()) {
            long topK = Integer.toUnsignedLong(sampling.getTopK());
            if (topK > Integer.MAX_VALUE) {
                throw GrpcFailures.invalid("sampling.top_k must fit a 32-bit integer, got " + topK);
            }
            sampler.topK((int) topK);
        }
        if (sampling.hasTopP()) {
            sampler.topP(finite("sampling.top_p", sampling.getTopP()));
        }
        if (sampling.hasMinP()) {
            sampler.minP(finite("sampling.min_p", sampling.getMinP()));
        }
        if (sampling.hasTypicalP()) {
            sampler.typicalP(finite("sampling.typical_p", sampling.getTypicalP()));
        }
        if (sampling.hasMinKeep()) {
            sampler.minKeep(sampling.getMinKeep());
        }
        if (sampling.hasSeed()) {
            long seed = sampling.getSeed();
            if (seed < SamplerConfig.RANDOM_SEED) {
                throw GrpcFailures.invalid("sampling.seed must be " + SamplerConfig.RANDOM_SEED
                        + " (random) or a non-negative seed, got " + seed);
            }
            sampler.seed(seed);
        }
        DecodingParameters decoding = request.getDecoding();
        if (decoding.hasPresencePenalty()) {
            sampler.presencePenalty(finite("decoding.presence_penalty", decoding.getPresencePenalty()));
        }
        if (decoding.hasFrequencyPenalty()) {
            sampler.frequencyPenalty(finite("decoding.frequency_penalty", decoding.getFrequencyPenalty()));
        }
        if (decoding.hasRepetitionPenalty()) {
            float repetition = finite("decoding.repetition_penalty", decoding.getRepetitionPenalty());
            if (repetition == 0f) {
                repetition = 1f; // vLLM's 0 means no penalty; the engine spells that as 1
            }
            sampler.repeatPenalty(repetition);
        }
        if (decoding.hasPenaltyLastN()) {
            sampler.penaltyLastN(decoding.getPenaltyLastN());
        }
        try {
            builder.sampler(sampler.build());
        } catch (IllegalArgumentException e) {
            throw GrpcFailures.invalid(e.getMessage());
        }
    }

    private static void stopping(GenerationRequest.Builder builder, StoppingCriteria stopping, int vocab) {
        if (stopping.getMaxNewTokens() > 0) {
            long max = Integer.toUnsignedLong(stopping.getMaxNewTokens());
            if (max > Integer.MAX_VALUE) {
                throw GrpcFailures.invalid("stopping.max_new_tokens must fit a 32-bit integer, got " + max);
            }
            builder.maxNewTokens((int) max);
        }
        if (stopping.getMinNewTokens() > 0) {
            long min = Integer.toUnsignedLong(stopping.getMinNewTokens());
            if (min > Integer.MAX_VALUE) {
                throw GrpcFailures.invalid("stopping.min_new_tokens must fit a 32-bit integer, got " + min);
            }
            builder.minTokens((int) min);
        }
        if (stopping.getStopTokenIdsCount() > 0) {
            for (int id : stopping.getStopTokenIdsList()) {
                if (Integer.toUnsignedLong(id) >= vocab) {
                    throw GrpcFailures.invalid("stopping.stop_token_ids: token id " + Integer.toUnsignedLong(id)
                            + " is outside the vocabulary [0, " + vocab + ")");
                }
            }
            builder.stopTokenIds(stopping.getStopTokenIdsList());
        }
        if (stopping.getStopStringsCount() > 0) {
            for (String stop : stopping.getStopStringsList()) {
                if (stop.isEmpty()) {
                    throw GrpcFailures.invalid("stopping.stop_strings must not contain an empty string");
                }
            }
            builder.stopStrings(stopping.getStopStringsList());
        }
        builder.includeStopString(stopping.getIncludeStopStrings());
        builder.ignoreEos(stopping.getIgnoreEos());
        if (stopping.hasThinkingTokenBudget()) {
            long budget = stopping.getThinkingTokenBudget();
            if (budget < GenerationRequest.REASONING_BUDGET_DISABLED || budget > Integer.MAX_VALUE) {
                throw GrpcFailures.invalid("stopping.thinking_token_budget must be "
                        + GenerationRequest.REASONING_BUDGET_DISABLED + " (disabled) or a non-negative token count,"
                        + " got " + budget);
            }
            builder.reasoningBudget((int) budget);
        }
    }

    // ------------------------------------------------------------------ decoding

    private static void decoding(GenerationRequest.Builder builder, GenerateRequest request, int vocab) {
        DecodingParameters decoding = request.getDecoding();
        if (decoding.getLogitBiasCount() > 0) {
            for (Map.Entry<Integer, Float> entry : decoding.getLogitBiasMap().entrySet()) {
                long id = Integer.toUnsignedLong(entry.getKey());
                if (id >= vocab) {
                    throw GrpcFailures.invalid("decoding.logit_bias token id " + id
                            + " is outside the vocabulary [0, " + vocab + ")");
                }
                try {
                    builder.logitBias(entry.getKey(), finite("decoding.logit_bias", entry.getValue()));
                } catch (IllegalArgumentException e) {
                    throw GrpcFailures.invalid(e.getMessage());
                }
            }
        }
        if (decoding.getAllowedTokenIdsCount() > 0) {
            for (int id : decoding.getAllowedTokenIdsList()) {
                if (Integer.toUnsignedLong(id) >= vocab) {
                    throw GrpcFailures.invalid("decoding.allowed_token_ids: token id " + Integer.toUnsignedLong(id)
                            + " is outside the vocabulary [0, " + vocab + ")");
                }
            }
            builder.allowedTokenIds(decoding.getAllowedTokenIdsList());
        }
        if (decoding.getBadWordsCount() > 0) {
            for (String word : decoding.getBadWordsList()) {
                if (word.isEmpty()) {
                    throw GrpcFailures.invalid("decoding.bad_words must not contain an empty string");
                }
            }
            builder.badWords(decoding.getBadWordsList());
        }
    }

    private static ConstraintSpec constraint(DecodingParameters decoding) {
        switch (decoding.getStructuredOutputCase()) {
            case JSON:
                return ConstraintSpec.jsonSchema(decoding.getJson(), null);
            case REGEX:
                return ConstraintSpec.regex(decoding.getRegex(), false);
            case CHOICE:
                if (decoding.getChoice().getChoicesCount() == 0) {
                    throw GrpcFailures.invalid("decoding.choice must name at least one choice");
                }
                return ConstraintSpec.choice(decoding.getChoice().getChoicesList());
            case GRAMMAR:
                return ConstraintSpec.grammar(decoding.getGrammar());
            case JSON_OBJECT:
                return decoding.getJsonObject() ? ConstraintSpec.jsonObject(null) : null;
            case STRUCTURAL_TAG:
                return ConstraintSpec.structuralTag(decoding.getStructuralTag());
            default:
                return null;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static void requireAdapter(EngineService service, String name) {
        for (LoraCatalog.Adapter adapter : service.loras().list()) {
            if (adapter.name().equals(name)) {
                return;
            }
        }
        throw GrpcFailures.invalid("lora_name: no LoRA adapter named '" + name + "' is loaded");
    }

    /** The cache salt, validated as the HTTP surfaces validate {@code cache_salt}. */
    private static String salt(String salt) {
        if (salt.isEmpty()) {
            return null;
        }
        boolean forbidden = false;
        for (int i = 0; i < salt.length() && !forbidden; i++) {
            char c = salt.charAt(i);
            forbidden = c == '@' || c == '/' || c == '\\' || c == '\0';
        }
        if (salt.length() > OpenAiRequests.MAX_CACHE_SALT || forbidden) {
            throw GrpcFailures.invalid("kv.cache_salt must be at most " + OpenAiRequests.MAX_CACHE_SALT
                    + " characters and must not contain '@', '/', '\\', or NUL");
        }
        return salt;
    }

    private static float finite(String field, float value) {
        if (!Float.isFinite(value)) {
            throw GrpcFailures.invalid(field + " must be finite, got " + value);
        }
        return value;
    }

    /**
     * The logprob selections of the response options: the output capture count (sampled token included), the output
     * {@code logprob_token_ids} rows, the prompt capture count and the prompt token-id rows. Output candidates beyond
     * the engine's alternative cap and the {@code all} output selection are refused by name.
     */
    private static final class Logprobs {
        private int outputEntries;
        private int[] outputTokenIds;
        private int promptEntries;
        private int[][] promptTokenIds;
        private boolean outputCandidates;

        static Logprobs of(ResponseOptions options, int vocab, int promptTokens) {
            Logprobs out = new Logprobs();
            if (options.hasOutputCandidates()) {
                CandidateTokens candidates = options.getOutputCandidates();
                switch (candidates.getSelectCase()) {
                    case TOP_N: {
                        int top = (int) Integer.toUnsignedLong(candidates.getTopN());
                        if (top > OpenAiRequests.MAX_TOP_LOGPROBS) {
                            throw GrpcFailures.invalid("response.output_candidates.top_n must be at most "
                                    + OpenAiRequests.MAX_TOP_LOGPROBS + ", got " + top);
                        }
                        out.outputCandidates = true;
                        out.outputEntries = top + 1;
                        break;
                    }
                    case TOKEN_IDS:
                        out.outputCandidates = true;
                        out.outputTokenIds = ids(candidates.getTokenIds(), vocab,
                                "response.output_candidates.token_ids");
                        break;
                    case ALL:
                        throw GrpcFailures.invalid("response.output_candidates.all is not supported; name top_n or"
                                + " token_ids");
                    default:
                        break;
                }
            }
            if (options.getOutputLogprobs() && out.outputEntries == 0 && out.outputTokenIds == null) {
                out.outputEntries = 1;
            }
            if (options.getPromptLogprobs()) {
                out.promptEntries = 1;
            }
            if (options.hasPromptCandidates()) {
                CandidateTokens candidates = options.getPromptCandidates();
                switch (candidates.getSelectCase()) {
                    case TOP_N: {
                        int top = (int) Integer.toUnsignedLong(candidates.getTopN());
                        if (top > OpenAiRequests.MAX_TOP_LOGPROBS) {
                            throw GrpcFailures.invalid("response.prompt_candidates.top_n must be at most "
                                    + OpenAiRequests.MAX_TOP_LOGPROBS + ", got " + top);
                        }
                        out.promptEntries = top + 1;
                        break;
                    }
                    case TOKEN_IDS: {
                        int[] wanted = ids(candidates.getTokenIds(), vocab,
                                "response.prompt_candidates.token_ids");
                        if (promptTokens < 2) {
                            throw GrpcFailures.invalid("response.prompt_candidates.token_ids needs a prompt of at"
                                    + " least two tokens, got " + promptTokens);
                        }
                        int[][] rows = new int[promptTokens - 1][];
                        for (int i = 0; i < rows.length; i++) {
                            rows[i] = wanted.clone();
                        }
                        out.promptTokenIds = rows;
                        break;
                    }
                    case ALL:
                        out.promptEntries = GenerationRequest.PROMPT_LOGPROBS_ALL;
                        break;
                    default:
                        break;
                }
            }
            return out;
        }
    }
}
