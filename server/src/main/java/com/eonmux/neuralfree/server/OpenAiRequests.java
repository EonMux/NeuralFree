package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps OpenAI-compatible request bodies onto the engine: the sampling settings (only present keys leave the
 * llama.cpp defaults), the number of choices, the grammar constraint in each of its spellings, the prompts of a
 * completion and the chat request with its tools. Every malformed field is a 400 that names it.
 */
final class OpenAiRequests {

    /** The most choices one request may ask for. */
    static final int MAX_CHOICES = 128;

    /** The most generations one request may queue: its prompts times {@code n}. */
    static final int MAX_GENERATIONS = 1024;

    /** The most alternatives a response lists per token (OpenAI's {@code top_logprobs} bound). */
    static final int MAX_TOP_LOGPROBS = 20;

    /** The most ids one {@code logprob_token_ids} list may name (vLLM's {@code MAX_LOGPROB_TOKEN_IDS}). */
    static final int MAX_LOGPROB_TOKEN_IDS = GenerationRequest.MAX_LOGPROB_TOKEN_IDS;

    /** The most characters a {@code cache_salt} may hold (vLLM's bound). */
    static final int MAX_CACHE_SALT = 128;

    /** The characters a {@code cache_salt} must not contain (vLLM's: they delimit keys in cache backends). */
    private static final String CACHE_SALT_FORBIDDEN = "@/\\\0";

    /** The {@code --grp-attn-n} launch default of {@code grp_attn_n} (llama.cpp's flag as a system property). */
    static final String PROP_GRP_ATTN_N = "neuralfree.grp-attn-n";

    /** The {@code --grp-attn-w} launch default of {@code grp_attn_w} (llama.cpp's flag as a system property). */
    static final String PROP_GRP_ATTN_W = "neuralfree.grp-attn-w";

    /** The {@code --cache-reuse} launch default of {@code n_cache_reuse} (llama.cpp's flag as a system property). */
    static final String PROP_CACHE_REUSE = "neuralfree.cache-reuse";

    private OpenAiRequests() {
    }

    /**
     * One completion prompt: text, token ids, or precomputed embedding rows — exactly one of the three
     * (vLLM's {@code prompt} and {@code prompt_embeds}); text may carry llama.cpp's {@code multimodal_data}, the
     * base64 media its markers stand for.
     *
     * @param text           the prompt text, or null
     * @param tokens         the prompt token ids, or null
     * @param embeds         the prompt embedding rows, or null
     * @param multimodalData the {@code multimodal_data} entries of a {@code prompt_string} object, or null
     * @param key            the request field the prompt came from, named in errors
     */
    record Prompt(String text, int[] tokens, float[][] embeds, List<?> multimodalData, String key) {
        Prompt {
            int given = (text != null ? 1 : 0) + (tokens != null ? 1 : 0) + (embeds != null ? 1 : 0);
            if (given != 1) {
                throw new IllegalArgumentException("a prompt takes text, token ids or embedding rows, exactly"
                        + " one of the three, got " + given);
            }
            if (multimodalData != null && text == null) {
                throw new IllegalArgumentException("multimodal_data belongs to a prompt_string");
            }
        }

        Prompt(String text) {
            this(text, null, null, null, "prompt");
        }

        Prompt(int[] tokens) {
            this(null, tokens, null, null, "prompt");
        }

        Prompt(float[][] embeds) {
            this(null, null, embeds, null, "prompt_embeds");
        }
    }

    /** {@code n}: the choices per prompt, in [1, {@link #MAX_CHOICES}]. */
    static int choices(JsonView body) {
        int n = body.optI32("n", 1);
        if (n < 1 || n > MAX_CHOICES) {
            throw ServerJson.badRequest("n must be in [1, " + MAX_CHOICES + "], got " + n);
        }
        return n;
    }

    /**
     * The alternatives a response lists per token, or -1 without logprobs: chat's boolean {@code logprobs} with
     * {@code top_logprobs}, the completions' integer {@code logprobs}, and llama.cpp's {@code n_probs}.
     */
    static int topLogprobs(JsonView body, boolean chat) {
        Object raw = body.raw("logprobs");
        int top = -1;
        if (Boolean.TRUE.equals(raw)) {
            top = body.optI32("top_logprobs", 0);
        } else if (!chat && (raw instanceof Long || raw instanceof Integer)) {
            top = ((Number) raw).intValue();
        } else if (raw != null && !(raw instanceof Boolean)) {
            throw ServerJson.badRequest("logprobs must be " + (chat ? "a boolean" : "an integer or a boolean"));
        } else if (chat && body.raw("top_logprobs") != null) {
            throw ServerJson.badRequest("top_logprobs needs logprobs: true");
        }
        int nProbs = body.optI32("n_probs", 0);
        if (nProbs > 0) {
            top = Math.max(top, nProbs);
        }
        if (top < -1 || top > MAX_TOP_LOGPROBS) {
            throw ServerJson.badRequest("the logprobs alternatives must be in [0, " + MAX_TOP_LOGPROBS + "], got "
                    + top);
        }
        return top;
    }

    /**
     * The generation settings of the body (the choices share them; {@link #forChoice} varies the seed). With
     * {@code topLogprobs} at 0 or above every token captures itself and {@code topLogprobs} alternatives. Carries
     * the vLLM extras the engine runs or reads back: the token controls of {@link TokenControls} (every token id
     * validated against {@code vocabSize}), {@code prompt_logprobs} (with the echo default),
     * {@code logprob_token_ids}, {@code priority}, {@code cache_salt}, llama.cpp's {@code n_keep}/{@code n_discard},
     * {@code grp_attn_n}/{@code grp_attn_w} and {@code n_cache_reuse}, the custom logits processor arguments of
     * {@code vllm_xargs} (validated against {@code processors}), and {@code kv_transfer_params} (a disaggregated
     * prefill/decode transfer). Logprobs score the default
     * {@link LogprobsMode#RAW_LOGPROBS}.
     */
    static GenerationRequest sampling(JsonView body, int topLogprobs, int vocabSize, boolean echo) {
        return sampling(body, topLogprobs, vocabSize, echo ? topLogprobs : -1, LogprobsMode.RAW_LOGPROBS);
    }

    /**
     * {@link #sampling(JsonView, int, int, boolean)} scoring logprobs and prompt logprobs in {@code mode}, the
     * echo default filling an absent {@code prompt_logprobs} with {@code echoTop} alternatives (-1 for none):
     * a completion's {@code logprobs}, a chat's {@code top_logprobs}.
     */
    static GenerationRequest sampling(JsonView body, int topLogprobs, int vocabSize, int echoTop,
                                      LogprobsMode mode) {
        return sampling(body, topLogprobs, vocabSize, echoTop, mode, LogitsProcessors.empty());
    }

    /**
     * {@link #sampling(JsonView, int, int, int, LogprobsMode)} with the engine's custom logits processors: their
     * factories validate the request's {@code vllm_xargs} arguments now, so an argument one refuses answers a 400
     * naming it before the request queues.
     */
    static GenerationRequest sampling(JsonView body, int topLogprobs, int vocabSize, int echoTop,
                                      LogprobsMode mode, LogitsProcessors processors) {
        return sampling(body, topLogprobs, vocabSize, echoTop, mode, processors, null);
    }

    /**
     * {@link #sampling(JsonView, int, int, int, LogprobsMode, LogitsProcessors)} with the engine's text watermark
     * ({@code engineWatermark}, null when the server serves none): the request's {@code watermarking} field turns it
     * off, and asking for it while the engine serves none is a 400 naming the missing configuration.
     */
    static GenerationRequest sampling(JsonView body, int topLogprobs, int vocabSize, int echoTop,
                                      LogprobsMode mode, LogitsProcessors processors,
                                      WatermarkConfig engineWatermark) {
        GenerationRequest.Builder builder = GenerationRequest.builder().logprobsMode(mode)
                .kvTransfer(kvTransfer(body));
        int maxTokens = body.has("max_completion_tokens") ? body.optI32("max_completion_tokens", -1)
                : body.optI32("max_tokens", -1);
        if (maxTokens >= 0) {
            builder.maxNewTokens(maxTokens);
        }
        if (topLogprobs >= 0) {
            builder.logprobs(topLogprobs + 1);
        }
        builder.stopStrings(strings(body, "stop"));
        builder.badWords(strings(body, "bad_words"));
        builder.minTokens(Math.max(0, body.optI32("min_tokens", 0)));
        if (body.optBool("ignore_eos", false)) {
            builder.ignoreEos(true);
        }
        ReasoningBudgets.apply(body, builder);
        Map<Integer, Float> bias = logitBias(body);
        if (!bias.isEmpty()) {
            builder.logitBias(bias);
        }
        TokenControls.apply(body, builder, vocabSize);
        builder.nKeep(nKeep(body));
        builder.nDiscard(nDiscard(body));
        int grpN = grpAttnN(body);
        int grpW = grpAttnW(body);
        if (grpN > GenerationRequest.GRP_ATTN_OFF && grpW % grpN != 0) {
            throw ServerJson.badRequest("grp_attn_w must be a multiple of grp_attn_n (llama.cpp:"
                    + " ga_w % ga_n == 0), got " + grpW + " / " + grpN);
        }
        builder.grpAttnN(grpN).grpAttnW(grpW);
        builder.nCacheReuse(cacheReuse(body));
        int promptLogprobs = echoPromptLogprobs(promptLogprobs(body), echoTop >= 0, echoTop);
        if (promptLogprobs != 0) {
            builder.promptLogprobs(promptLogprobs);
        }
        int[] logprobTokenIds = logprobTokenIds(body, vocabSize, topLogprobs);
        if (logprobTokenIds.length > 0) {
            builder.logprobTokenIds(logprobTokenIds);
        }
        builder.priority(priority(body));
        builder.cacheSalt(cacheSalt(body));
        LogitsProcessorArgs args = logitsProcessorArgs(body);
        try {
            processors.validate(args);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        if (!args.isEmpty()) {
            builder.logitsProcessorArgs(args);
        }
        builder.sampler(SamplingParams.of(body, engineWatermark));
        return builder.build();
    }

    /**
     * The body's {@code cache_salt}, or null when absent or null (vLLM's {@code validate_cache_salt}): a non-empty
     * string of at most {@link #MAX_CACHE_SALT} characters without {@code @}, {@code /}, {@code \} or NUL. The
     * engine mixes it into the hash of the first prefix-cache block, so only requests of the same salt share
     * cached blocks.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code cache_salt} for any other value
     */
    static String cacheSalt(JsonView body) {
        Object raw = body.raw("cache_salt");
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof String salt) || salt.isEmpty()) {
            throw ServerJson.badRequest("Parameter 'cache_salt' must be a non-empty string if provided.");
        }
        if (salt.length() > MAX_CACHE_SALT || salt.chars().anyMatch(c -> CACHE_SALT_FORBIDDEN.indexOf(c) >= 0)) {
            throw ServerJson.badRequest("Parameter 'cache_salt' must be at most " + MAX_CACHE_SALT
                    + " characters and must not contain '@', '/', '\\', or NUL.");
        }
        return salt;
    }

    /**
     * The response id a request names: the {@code X-Request-Id} header first (vLLM's {@code _base_request_id},
     * which the generation and pooling routes read), then the body's {@code request_id}; null when neither names
     * one. A null request (a resource built outside the HTTP layer) reads the body alone.
     */
    static String requestId(HttpServerRequest http, JsonView body) {
        if (http != null) {
            String named = http.headers().get("X-Request-Id");
            if (named != null && !named.isEmpty()) {
                return named;
            }
        }
        return body == null ? null : requestId(body);
    }

    /** The request body's {@code request_id}, or null when absent; anything but a string refuses by name. */
    static String requestId(JsonView body) {
        if (body.raw("request_id") == null) {
            return null;
        }
        if (!(body.raw("request_id") instanceof String s)) {
            throw ServerJson.badRequest("request_id must be a string");
        }
        return s;
    }

    /** The request's {@code use_activation} (vLLM's pooler flag): the head's activation, true by default. */
    static boolean useActivation(JsonView body) {
        return body.optBool("use_activation", true);
    }

    /**
     * The request's {@code kv_transfer_params} (vLLM's disaggregated prefill/decode field), or null when absent.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming an unknown, mistyped or contradictory field
     */
    static KvTransferParams kvTransfer(JsonView body) {
        if (body.raw(KvTransferParams.FIELD) == null) {
            return null;
        }
        try {
            return KvTransferParams.parse(body.obj(KvTransferParams.FIELD));
        } catch (JsonException e) {
            throw ServerJson.badRequest(String.valueOf(e.getMessage()));
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /**
     * Refuses a {@code kv_transfer_params} that is not an object: the routes that carry the field into an OpenAI
     * body (Cohere's) validate the shape where they convert, and the full parse runs when the request is planned.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field when it is not an object
     */
    static void requireKvTransferObject(JsonView body) {
        Object raw = body.raw(KvTransferParams.FIELD);
        if (raw != null && !(raw instanceof Map<?, ?>)) {
            throw ServerJson.badRequest(KvTransferParams.FIELD + " must be an object");
        }
    }

    /**
     * Refuses a disaggregated-transfer request that asks for more than one generation: a transfer moves one
     * prompt's cached rows under one transfer id, so the prefill and decode sides must agree on it.
     *
     * @param request     the sampling request (its {@code kvTransfer()} decides)
     * @param prompts     the request's prompts
     * @param choices     {@code n}
     * @param candidates  {@code best_of}
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the counts
     */
    static void requireSingleTransfer(GenerationRequest request, int prompts, int choices, int candidates) {
        if (request.kvTransfer() == null || (prompts == 1 && choices == 1 && candidates == 1)) {
            return;
        }
        throw ServerJson.badRequest("kv_transfer_params serves one prompt and one choice: send one prompt with n"
                + " and best_of at 1, got " + prompts + " prompt(s), n " + choices + ", best_of " + candidates);
    }

    /**
     * The body's {@code vllm_xargs} object as a request's custom logits processor arguments (vLLM's
     * {@code SamplingParams.extra_args}); {@link LogitsProcessorArgs#empty()} when absent or null. The values are a
     * string, boolean, integer, number or a flat array of those, each checked here.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code vllm_xargs} or its member for any other shape
     */
    static LogitsProcessorArgs logitsProcessorArgs(JsonView body) {
        Object raw = body.raw("vllm_xargs");
        if (raw == null) {
            return LogitsProcessorArgs.empty();
        }
        if (!(raw instanceof Map<?, ?> map)) {
            throw ServerJson.badRequest("vllm_xargs must be an object of argument names to values");
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            if (!(e.getKey() instanceof String name)) {
                throw ServerJson.badRequest("vllm_xargs must be an object of argument names to values");
            }
            values.put(name, e.getValue());
        }
        try {
            return LogitsProcessorArgs.of(values);
        } catch (IllegalArgumentException ex) {
            throw ServerJson.badRequest(ex.getMessage());
        }
    }

    /**
     * llama.cpp's {@code n_keep}: the prompt tokens a context shift keeps, -1 keeping the whole prompt. Absent is
     * 0 (a leading BOS only); anything else below -1 refuses by name.
     */
    static int nKeep(JsonView body) {
        long raw = body.optI64("n_keep", 0);
        if (raw < GenerationRequest.KEEP_PROMPT_ALL || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("n_keep must be -1 (the whole prompt) or a non-negative token count, got "
                    + raw);
        }
        return (int) raw;
    }

    /**
     * llama.cpp's {@code n_discard}: the tokens one context shift discards after {@code n_keep}. Absent is 0 (the
     * default half of what lies beyond the kept tokens); a negative count refuses by name.
     */
    static int nDiscard(JsonView body) {
        long raw = body.optI64("n_discard", 0);
        if (raw < 0 || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("n_discard must be a non-negative token count, 0 taking the default"
                    + " half, got " + raw);
        }
        return (int) raw;
    }

    /**
     * llama.cpp's {@code --grp-attn-n} as a request field: the self-extend group-attention factor, 1 (the default)
     * leaving every position as stored. Anything below 1 refuses by name.
     */
    static int grpAttnN(JsonView body) {
        long raw = body.optI64("grp_attn_n", grpAttnNDefault());
        if (raw < GenerationRequest.GRP_ATTN_OFF || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("grp_attn_n must be 1 (self-extend off) or a positive group factor, got "
                    + raw);
        }
        return (int) raw;
    }

    /**
     * llama.cpp's {@code --grp-attn-w} as a request field: the positions one self-extend group spans, llama.cpp's
     * {@value GenerationRequest#DEFAULT_GRP_ATTN_W} by default and a multiple of {@code grp_attn_n}. A value below
     * 1 refuses by name.
     */
    static int grpAttnW(JsonView body) {
        long raw = body.optI64("grp_attn_w", grpAttnWDefault());
        if (raw < 1 || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("grp_attn_w must be a positive token count, got " + raw);
        }
        return (int) raw;
    }

    /**
     * llama.cpp's {@code n_cache_reuse} as a request field ({@code --cache-reuse}): the shortest cached run past
     * the prompt's common prefix that a KV chunk reuse re-appends at the prompt's own rows, 0 leaving the reuse
     * off. Absent reads the launch default ({@code --cache-reuse}, 0 — llama.cpp's); a negative count or anything
     * but a whole number refuses by name.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code n_cache_reuse}
     */
    static int cacheReuse(JsonView body) {
        long raw = body.optI64("n_cache_reuse", cacheReuseDefault());
        if (raw < 0 || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("n_cache_reuse must be a non-negative token count, 0 turning the chunk"
                    + " reuse off, got " + raw);
        }
        return (int) raw;
    }

    /** The {@code --grp-attn-n} launch default of {@code grp_attn_n}, 1 when the launch names none. */
    static int grpAttnNDefault() {
        return launchDefault(PROP_GRP_ATTN_N, 1);
    }

    /** The {@code --grp-attn-w} launch default of {@code grp_attn_w}, llama.cpp's when the launch names none. */
    static int grpAttnWDefault() {
        return launchDefault(PROP_GRP_ATTN_W, GenerationRequest.DEFAULT_GRP_ATTN_W);
    }

    /** The {@code --cache-reuse} launch default of {@code n_cache_reuse}, 0 (llama.cpp's) when the launch names none. */
    static int cacheReuseDefault() {
        return launchDefault(PROP_CACHE_REUSE, 0);
    }

    /** The launch default named by {@code property}, validated like its request field, or {@code fallback}. */
    private static int launchDefault(String property, int fallback) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        long value;
        try {
            value = Long.parseLong(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + " must be a whole number, got '" + raw + "'");
        }
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(property + " must be a non-negative token count, got " + value);
        }
        return (int) value;
    }

    /**
     * vLLM's {@code best_of}: the candidates generated per prompt, from which the best {@code n} by cumulative
     * log probability are returned. Absent means {@code n}; a value below {@code n}, above
     * {@link #MAX_CHOICES} or beside a stream refuses by name (vLLM does not stream when {@code best_of > n}).
     */
    static int bestOf(JsonView body, int n, boolean streamed) {
        Object raw = body.raw("best_of");
        if (raw == null) {
            return n;
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw ServerJson.badRequest("best_of must be an integer");
        }
        long v = ((Number) raw).longValue();
        if (v < n) {
            throw ServerJson.badRequest("best_of must be greater than or equal to n (" + n + "), got " + v);
        }
        if (v > MAX_CHOICES) {
            throw ServerJson.badRequest("best_of must be in [n, " + MAX_CHOICES + "], got " + v);
        }
        if (streamed && v > n) {
            throw ServerJson.badRequest("best_of " + v + " with n " + n + " cannot serve a stream: the best"
                    + " choices are only known once every candidate finished");
        }
        return (int) v;
    }

    /**
     * vLLM's {@code logprob_token_ids}: the vocabulary ids whose log probability every generated token's capture
     * reports, the sampled token leading (they stand in for the top alternatives). Absent, null or empty means
     * none, as vLLM reads it. An id outside {@code vocabSize}, more than {@link #MAX_LOGPROB_TOKEN_IDS} ids or a
     * request without {@code logprobs} refuses by name.
     */
    static int[] logprobTokenIds(JsonView body, int vocabSize, int top) {
        Object raw = body.raw("logprob_token_ids");
        if (raw == null || raw instanceof List<?> empty && empty.isEmpty()) {
            return new int[0];
        }
        if (top < 0) {
            throw ServerJson.badRequest("logprob_token_ids needs logprobs to be set");
        }
        int[] ids = TokenControls.ids(raw, "logprob_token_ids", vocabSize, false);
        if (ids.length > MAX_LOGPROB_TOKEN_IDS) {
            throw ServerJson.badRequest("logprob_token_ids may name at most " + MAX_LOGPROB_TOKEN_IDS
                    + " token ids, got " + ids.length);
        }
        return ids;
    }

    /**
     * The sampling request every {@code best_of} candidate runs: the candidates are ranked by cumulative log
     * probability, so with more candidates than answers each one captures its sampled tokens' scores even when the
     * request asked for none (the response renders them only when it did). The ranking capture is the request's
     * own {@code LogprobCapture}, scored in its {@link GenerationRequest#logprobsMode()} as vLLM ranks.
     */
    static GenerationRequest scoredCandidates(GenerationRequest base, int n, int bestOf) {
        return bestOf > n && !base.wantsLogprobs() ? base.toBuilder().logprobs(1).build() : base;
    }

    /**
     * vLLM's {@code use_beam_search}: the request answers through beam search, whose beams are the request's
     * {@code n} (vLLM's {@code to_beam_search_params} maps {@code beam_width = n}). Absent or false means
     * sampling; a non-boolean refuses by name.
     */
    static boolean beamSearch(JsonView body) {
        Object raw = body.raw("use_beam_search");
        if (raw == null) {
            return false;
        }
        if (!(raw instanceof Boolean)) {
            throw ServerJson.badRequest("use_beam_search must be a boolean, got " + raw);
        }
        return (Boolean) raw;
    }

    /**
     * vLLM's {@code length_penalty}: the exponent the beam ranking divides the cumulative log probability by
     * ({@code len(logprobs) ^ length_penalty}). Absent means 1.0 (vLLM's default); a non-number or a
     * non-finite value refuses by name.
     */
    static float lengthPenalty(JsonView body) {
        Object raw = body.raw("length_penalty");
        if (raw == null) {
            return 1.0f;
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw ServerJson.badRequest("length_penalty must be a number, got " + raw);
        }
        float v = ((Number) raw).floatValue();
        if (Float.isNaN(v) || Float.isInfinite(v)) {
            throw ServerJson.badRequest("length_penalty must be finite, got " + raw);
        }
        return v;
    }

    /**
     * The beam-search settings onto the sampling request when {@code use_beam_search} is set: the search
     * expands over {@code n} beams ranked by the length-penalized score. Beside {@code kv_transfer_params} it
     * refuses: the transfer's rows come from the batch engine the search does not run.
     */
    static GenerationRequest withBeamSearch(GenerationRequest base, JsonView body, int n) {
        if (!beamSearch(body)) {
            return base;
        }
        if (base.kvTransfer() != null) {
            throw ServerJson.badRequest("use_beam_search ranks its beams outside the batch engine, so it cannot"
                    + " run a kv_transfer_params transfer; drop one of the two fields");
        }
        return base.toBuilder().beamWidth(n).lengthPenalty(lengthPenalty(body)).build();
    }

    /**
     * Refuses a stream whose explicit {@code prompt_logprobs} asks for alternatives or every token (vLLM's rule:
     * the echo default does not count, and {@code prompt_logprobs: 0} streams).
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code prompt_logprobs}
     */
    static void requireStreamablePromptLogprobs(JsonView body) {
        int explicit = promptLogprobs(body);
        if (explicit > 1 || explicit == GenerationRequest.PROMPT_LOGPROBS_ALL) {
            throw ServerJson.badRequest("prompt_logprobs are not available when stream is true");
        }
    }

    /**
     * vLLM's {@code prompt_logprobs} as the engine's per-position entry count: an integer of -1 (every
     * vocabulary token) or 0 and above (the prompt token plus that many alternatives), absent meaning off.
     */
    static int promptLogprobs(JsonView body) {
        Object raw = body.raw("prompt_logprobs");
        if (raw == null) {
            return 0;
        }
        if (!(raw instanceof Number) || raw instanceof Boolean) {
            throw ServerJson.badRequest("prompt_logprobs must be -1 or a non-negative integer");
        }
        long v = ((Number) raw).longValue();
        if (v == -1) {
            return GenerationRequest.PROMPT_LOGPROBS_ALL;
        }
        if (v < 0) {
            throw ServerJson.badRequest("prompt_logprobs must be -1 or a non-negative integer, got " + v);
        }
        return (int) Math.min(v + 1, Integer.MAX_VALUE);
    }

    /** Whether the echo default fills {@code prompt_logprobs} with the logprobs count (vLLM's rule). */
    static int echoPromptLogprobs(int promptLogprobs, boolean echo, int top) {
        return promptLogprobs != 0 || !echo ? promptLogprobs : Math.max(0, top + 1);
    }

    /**
     * vLLM's {@code truncate_prompt_tokens} with {@code truncation_side}: {@code tokens} cut to at most
     * {@code n}, keeping the first (side {@code right}, the default) or the last (side {@code left}) ones.
     */
    static int[] truncate(JsonView body, int[] tokens) {
        if (!body.has("truncate_prompt_tokens")) {
            return tokens;
        }
        long raw = body.optI64("truncate_prompt_tokens", -1);
        if (raw < 1) {
            throw ServerJson.badRequest("truncate_prompt_tokens must be a positive token count, got " + raw);
        }
        if (raw >= tokens.length) {
            return tokens;
        }
        String side = body.optStr("truncation_side", "right");
        if (!side.equals("left") && !side.equals("right")) {
            throw ServerJson.badRequest("truncation_side must be left or right, got '" + side + "'");
        }
        int n = (int) raw;
        return side.equals("left") ? Arrays.copyOfRange(tokens, tokens.length - n, tokens.length)
                : Arrays.copyOf(tokens, n);
    }

    /**
     * Refuses {@code request}'s {@code priority} when {@code policy} does not serve it (any value but 0 under
     * {@code fcfs}), as the batch engine would at submit; checked while a request is planned, so a streaming
     * request answers 400 instead of opening its stream.
     */
    static void requireServedPriority(GenerationRequest request, SchedulingPolicy policy) {
        String refusal = policy.priorityRefusal(request.priority());
        if (refusal != null) {
            throw ServerJson.badRequest(refusal);
        }
    }

    /** vLLM's {@code priority}: any 32-bit integer; a wider value refuses by name. */
    static int priority(JsonView body) {
        long raw = body.optI64("priority", 0);
        if (raw < Integer.MIN_VALUE || raw > Integer.MAX_VALUE) {
            throw ServerJson.badRequest("priority must fit a 32-bit integer, got " + raw);
        }
        return (int) raw;
    }

    /** {@code stream_options.continuous_usage_stats}: running usage on every chunk (needs {@code include_usage}). */
    static boolean continuousUsageStats(JsonView body) {
        return body.optBool("stream", false) && body.optObj("stream_options").optBool("include_usage", false)
                && body.optObj("stream_options").optBool("continuous_usage_stats", false);
    }

    /**
     * The request of choice {@code index}: with an explicit seed, choice {@code i} samples with {@code seed + i}
     * (vLLM's parallel sampling); without one every choice draws its own random seed.
     */
    static GenerationRequest forChoice(GenerationRequest base, int index) {
        long seed = base.sampler().seed();
        if (index == 0 || seed == SamplerConfig.RANDOM_SEED) {
            return base;
        }
        return base.toBuilder().sampler(base.sampler().withSeed(seed + index)).build();
    }

    private static Map<Integer, Float> logitBias(JsonView body) {
        Map<Integer, Float> out = new LinkedHashMap<>();
        if (!body.has("logit_bias")) {
            return out;
        }
        if (!(body.raw("logit_bias") instanceof Map<?, ?> map)) {
            throw ServerJson.badRequest("logit_bias must be an object of token id to bias");
        }
        for (Map.Entry<?, ?> e : map.entrySet()) {
            int id;
            try {
                id = Integer.parseInt(String.valueOf(e.getKey()).trim());
            } catch (NumberFormatException parse) {
                throw ServerJson.badRequest("logit_bias key '" + e.getKey() + "' is not a token id");
            }
            if (id < 0) {
                throw ServerJson.badRequest("logit_bias token id " + id + " is negative");
            }
            double bias = e.getValue() instanceof Number num ? num.doubleValue() : Double.NaN;
            if (!Double.isFinite(bias) || bias < -100.0 || bias > 100.0) {
                throw ServerJson.badRequest("logit_bias value of token " + id + " must be a number in [-100, 100]");
            }
            out.put(id, (float) bias);
        }
        return out;
    }

    private static List<String> strings(JsonView body, String key) {
        List<String> out = new ArrayList<>();
        Object raw = body.raw(key);
        if (raw instanceof String str) {
            out.add(str);
        } else if (raw instanceof List<?> list) {
            for (Object o : list) {
                if (!(o instanceof String str)) {
                    throw ServerJson.badRequest(key + " must be a string or an array of strings");
                }
                out.add(str);
            }
        } else if (raw != null) {
            throw ServerJson.badRequest(key + " must be a string or an array of strings");
        }
        return out;
    }

    // ------------------------------------------------------------------ constraints

    /**
     * The grammar constraint of the body, or null: llama.cpp's {@code grammar} and {@code json_schema}, the vLLM
     * {@code guided_*} fields and {@code structured_outputs}, and OpenAI's {@code response_format}. Two of them
     * at once are refused. A grammar field takes GBNF or Lark, told apart as vLLM does
     * ({@link ConstraintSpec#grammar}).
     */
    static ConstraintSpec constraint(JsonView body) {
        List<String> given = new ArrayList<>();
        ConstraintSpec spec = null;
        for (String key : new String[]{"grammar", "json_schema", "guided_json", "guided_regex", "guided_choice",
                "guided_grammar", "guided_structural_tag", "structural_tag", "response_format",
                "structured_outputs"}) {
            if (!body.has(key) || body.raw(key) == null) {
                continue;
            }
            ConstraintSpec c = constraintOf(body, key);
            if (c != null) {
                given.add(key);
                spec = c;
            }
        }
        if (given.size() > 1) {
            throw ServerJson.badRequest("only one output constraint may be given, got " + given);
        }
        return spec;
    }

    private static ConstraintSpec constraintOf(JsonView body, String key) {
        switch (key) {
            case "grammar":
            case "guided_grammar":
                return ConstraintSpec.grammar(body.str(key));
            case "json_schema":
            case "guided_json":
                return ConstraintSpec.jsonSchema(jsonText(body.raw(key), key), null);
            case "guided_regex":
                return ConstraintSpec.regex(body.str(key), false);
            case "guided_choice":
                return ConstraintSpec.choice(stringList(body.raw(key), key));
            case "guided_structural_tag":
            case "structural_tag":
                return ConstraintSpec.structuralTag(jsonText(body.raw(key), key));
            case "response_format":
                return responseFormat(body.obj(key), body.raw(key));
            default:
                return structuredOutputs(body.obj(key));
        }
    }

    /**
     * OpenAI's {@code response_format}: {@code text}, {@code json_object} or {@code json_schema}, or vLLM's
     * {@code structural_tag}, whose whole object (a {@code format} tree or the legacy {@code structures} and
     * {@code triggers}) is the structural-tag payload.
     */
    private static ConstraintSpec responseFormat(JsonView format, Object raw) {
        String type = format.optStr("type", "text");
        switch (type) {
            case "text":
                return null;
            case "json_object":
                return format.has("schema") ? ConstraintSpec.jsonSchema(jsonText(format.raw("schema"),
                        "response_format.schema"), null) : ConstraintSpec.jsonObject(null);
            case "json_schema": {
                JsonView spec = format.optObj("json_schema");
                Object schema = spec.has("schema") ? spec.raw("schema") : format.raw("schema");
                return schema == null ? ConstraintSpec.jsonObject(null)
                        : ConstraintSpec.jsonSchema(jsonText(schema, "response_format.json_schema.schema"), null);
            }
            case "structural_tag":
                return ConstraintSpec.structuralTag(jsonText(raw, "response_format"));
            default:
                throw ServerJson.badRequest("response_format type '" + type
                        + "' is not one of text, json_object, json_schema, structural_tag");
        }
    }

    /**
     * vLLM's {@code structured_outputs}: exactly one of json, regex, choice, grammar, json_object, structural_tag,
     * plus the JSON options {@code disable_any_whitespace}, {@code whitespace_pattern} and
     * {@code disable_additional_properties}, which ride along with a {@code json} or {@code json_object}
     * constraint ({@link ConstraintSpec#withJsonOptions}) and are refused beside any other kind.
     */
    private static ConstraintSpec structuredOutputs(JsonView so) {
        List<ConstraintSpec> specs = new ArrayList<>();
        if (so.has("json")) {
            specs.add(ConstraintSpec.jsonSchema(jsonText(so.raw("json"), "structured_outputs.json"), null));
        }
        if (so.has("regex")) {
            specs.add(ConstraintSpec.regex(so.str("regex"), false));
        }
        if (so.has("choice")) {
            specs.add(ConstraintSpec.choice(stringList(so.raw("choice"), "structured_outputs.choice")));
        }
        if (so.has("grammar")) {
            specs.add(ConstraintSpec.grammar(so.str("grammar")));
        }
        if (so.optBool("json_object", false)) {
            specs.add(ConstraintSpec.jsonObject(null));
        }
        if (so.has("structural_tag")) {
            specs.add(ConstraintSpec.structuralTag(jsonText(so.raw("structural_tag"),
                    "structured_outputs.structural_tag")));
        }
        if (specs.size() > 1) {
            throw ServerJson.badRequest("structured_outputs takes exactly one constraint kind");
        }
        boolean disableAnyWhitespace = so.optBool("disable_any_whitespace", false);
        String whitespacePattern = so.raw("whitespace_pattern") == null ? null : so.str("whitespace_pattern");
        boolean disableAdditionalProperties = so.optBool("disable_additional_properties", false);
        if (specs.isEmpty()) {
            if (disableAnyWhitespace || whitespacePattern != null || disableAdditionalProperties) {
                throw ServerJson.badRequest("structured_outputs needs a json or json_object constraint beside its"
                        + " disable_any_whitespace, whitespace_pattern and disable_additional_properties options");
            }
            return null;
        }
        return specs.get(0).withJsonOptions(disableAnyWhitespace, whitespacePattern,
                disableAdditionalProperties);
    }

    /** A JSON field given as an object or as JSON text, as text. */
    private static String jsonText(Object raw, String key) {
        if (raw instanceof String s) {
            return s;
        }
        if (raw instanceof Map<?, ?> || raw instanceof List<?>) {
            return ServerJson.write(raw);
        }
        throw ServerJson.badRequest(key + " must be a JSON object or JSON text");
    }

    private static List<String> stringList(Object raw, String key) {
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(key + " must be a non-empty array of strings");
        }
        List<String> out = new ArrayList<>();
        for (Object o : list) {
            if (!(o instanceof String s)) {
                throw ServerJson.badRequest(key + " must be a non-empty array of strings");
            }
            out.add(s);
        }
        return out;
    }

    /**
     * The output modalities of a chat completion ({@code modalities}, OpenAI's {@code output_modalities} alias):
     * a non-empty subset of {@code text} and {@code audio} in request order; {"text"} when neither field names
     * anything. Naming both fields refuses, as vLLM refuses a double-specified output shape; naming anything else
     * refuses by name.
     */
    static List<String> outputModalities(JsonView body) {
        Object raw = body.raw("modalities");
        Object openAiRaw = body.raw("output_modalities");
        if (raw != null && openAiRaw != null) {
            throw ServerJson.badRequest("name the output modalities once: \"modalities\" and"
                    + " \"output_modalities\" cannot both be set");
        }
        if (raw == null && openAiRaw == null) {
            return List.of("text");
        }
        String key = raw != null ? "modalities" : "output_modalities";
        List<String> named = stringList(raw != null ? raw : openAiRaw, key);
        for (String modality : named) {
            if (!"text".equals(modality) && !"audio".equals(modality)) {
                throw ServerJson.badRequest("\"" + key + "\" holds \"" + modality + "\", which is not one of"
                        + " \"text\" and \"audio\"");
            }
        }
        return named;
    }

    /**
     * The audio output settings of a chat request: OpenAI's {@code audio} object ({@code voice}, {@code format})
     * with llama.cpp's {@code lang} and {@code voice} /tts fields as fallbacks, so a request may name them either
     * way. Naming a field twice with different values refuses; a format other than {@code wav} or {@code pcm16}
     * refuses by name.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 for a malformed object, an unknown key, a conflict, or an
     *                                               unsupported format
     */
    static AudioOutput audioOutput(JsonView body) {
        String voice = null;
        String format = "wav";
        String language = body.optStr("lang", "");
        Object raw = body.raw("audio");
        if (raw != null) {
            if (!(raw instanceof Map<?, ?>)) {
                throw ServerJson.badRequest("\"audio\" must be an object with \"voice\" and \"format\"");
            }
            JsonView audio = JsonView.of(raw, "audio");
            for (String key : audio.keys()) {
                if (!"voice".equals(key) && !"format".equals(key)) {
                    throw ServerJson.badRequest("\"audio\" holds \"" + key + "\"; it takes \"voice\" and"
                            + " \"format\"");
                }
            }
            voice = audio.optStr("voice", "");
            format = audio.optStr("format", "wav");
        }
        String alias = body.optStr("voice", "");
        if (!alias.isEmpty()) {
            if (voice != null && !voice.isEmpty() && !voice.equals(alias)) {
                throw ServerJson.badRequest("name the voice once: \"audio.voice\" is \"" + voice + "\" and"
                        + " \"voice\" is \"" + alias + "\"");
            }
            voice = alias;
        }
        if (!"wav".equals(format) && !"pcm16".equals(format)) {
            throw ServerJson.badRequest("\"audio.format\" is \"" + format + "\", which this server does not"
                    + " encode; use \"wav\" or \"pcm16\"");
        }
        return new AudioOutput(voice == null ? "" : voice, format, language);
    }

    /** The audio output settings of a chat request ({@link #audioOutput}). */
    record AudioOutput(String voice, String format, String language) {
    }

    // ------------------------------------------------------------------ prompts

    /**
     * The prompts of a completion: vLLM's {@code prompt_embeds} first (embedding rows, one prompt per entry),
     * then OpenAI's four {@code prompt} forms (a string, an array of strings, an array of token ids, or an
     * array of token-id arrays). At least one prompt must come out.
     *
     * @param vocabSize the bound of a token id
     * @param embedWidth the model's embedding width, or 0 to leave row widths unchecked here
     */
    static List<Prompt> prompts(JsonView body, int vocabSize, int embedWidth) {
        List<Prompt> out = new ArrayList<>();
        Object embeds = body.raw("prompt_embeds");
        if (embeds instanceof List<?> list) {
            if (!list.isEmpty()) {
                out.addAll(embedPrompts(list, embedWidth));
            }
        } else if (embeds != null) {
            throw ServerJson.badRequest("prompt_embeds must be a non-empty array of embedding rows");
        }
        Object raw = body.raw("prompt");
        if (raw != null) {
            parsePrompt(raw, vocabSize, out);
        }
        if (out.isEmpty()) {
            throw ServerJson.badRequest("Either prompt or prompt_embeds must be provided and non-empty.");
        }
        return out;
    }

    /**
     * The prompts of a {@code prompt_embeds} array: a flat row (one prompt of one row), a list of rows (one
     * prompt), or a list of row lists (one prompt per entry), vLLM's {@code bytes | list[bytes]} shape.
     */
    private static List<Prompt> embedPrompts(Object raw, int width) {
        List<?> list = (List<?>) raw;
        List<Prompt> out = new ArrayList<>();
        if (list.get(0) instanceof Number) {
            out.add(new Prompt(new float[][] {embedRow(list, "prompt_embeds", width)}));
            return out;
        }
        if (list.get(0) instanceof List<?> first && !first.isEmpty() && first.get(0) instanceof List<?>) {
            for (int i = 0; i < list.size(); i++) {
                if (!(list.get(i) instanceof List<?> entry)) {
                    throw ServerJson.badRequest("prompt_embeds[" + i + "] must be an array of embedding rows");
                }
                out.add(new Prompt(embedRows(entry, "prompt_embeds[" + i + "]", width)));
            }
            return out;
        }
        out.add(new Prompt(embedRows(list, "prompt_embeds", width)));
        return out;
    }

    /** The embedding rows of one prompt; every row of the model's {@code embedWidth}. */
    private static float[][] embedRows(List<?> rows, String key, int width) {
        if (rows.isEmpty()) {
            throw ServerJson.badRequest(key + " must hold at least one embedding row");
        }
        float[][] out = new float[rows.size()][];
        for (int i = 0; i < out.length; i++) {
            if (!(rows.get(i) instanceof List<?> row)) {
                throw ServerJson.badRequest(key + "[" + i + "] must be an array of numbers");
            }
            out[i] = embedRow(row, key + "[" + i + "]", width);
        }
        return out;
    }

    /** One embedding row: finite numbers, of the model's {@code embedWidth} when it is known. */
    static float[] embedRow(List<?> values, String key, int width) {
        if (values.isEmpty()) {
            throw ServerJson.badRequest(key + " must hold at least one value");
        }
        float[] row = new float[values.size()];
        for (int i = 0; i < row.length; i++) {
            if (!(values.get(i) instanceof Number n)) {
                throw ServerJson.badRequest(key + " must hold numbers");
            }
            float v = n.floatValue();
            if (!Float.isFinite(v)) {
                throw ServerJson.badRequest(key + " must hold finite numbers, got " + values.get(i));
            }
            row[i] = v;
        }
        if (width > 0 && row.length != width) {
            throw ServerJson.badRequest(key + " width " + row.length + " does not match the model's embedding"
                    + " width " + width);
        }
        return row;
    }

    /** The four OpenAI {@code prompt} forms, appended to {@code out}. */
    private static void parsePrompt(Object raw, int vocabSize, List<Prompt> out) {
        if (raw instanceof String s) {
            out.add(new Prompt(s));
        } else if (raw instanceof Map<?, ?> object) {
            out.add(objectPrompt(object, "prompt"));
        } else if (raw instanceof List<?> list && !list.isEmpty()) {
            if (list.get(0) instanceof Number) {
                out.add(new Prompt(tokens(list, vocabSize, "prompt")));
            } else {
                for (int i = 0; i < list.size(); i++) {
                    Object item = list.get(i);
                    if (item instanceof String s) {
                        out.add(new Prompt(s));
                    } else if (item instanceof List<?> ids && !ids.isEmpty()) {
                        out.add(new Prompt(tokens(ids, vocabSize, "prompt[" + i + "]")));
                    } else if (item instanceof Map<?, ?> object) {
                        out.add(objectPrompt(object, "prompt[" + i + "]"));
                    } else {
                        throw ServerJson.badRequest("prompt[" + i + "] must be a string, a non-empty array of"
                                + " token ids or a prompt_string object");
                    }
                }
            }
        } else if (out.isEmpty()) {
            throw ServerJson.badRequest("Either prompt or prompt_embeds must be provided and non-empty.");
        }
    }

    /**
     * llama.cpp's prompt object, {@code {"prompt_string": ..., "multimodal_data": [base64, ...]}}
     * ({@code tokenize_input_subprompt}): the text, with the media its markers stand for when the data is given.
     */
    private static Prompt objectPrompt(Map<?, ?> object, String key) {
        if (!(object.get("prompt_string") instanceof String text)) {
            throw ServerJson.badRequest(key + " is an object without a string 'prompt_string'");
        }
        Object data = object.get("multimodal_data");
        if (data == null) {
            return new Prompt(text, null, null, null, key);
        }
        if (!(data instanceof List<?> entries) || entries.isEmpty()) {
            throw ServerJson.badRequest(key + ".multimodal_data must be a non-empty array of base64 strings");
        }
        return new Prompt(text, null, null, entries, key);
    }

    /** Refuses a request whose prompts times {@code n} exceed {@link #MAX_GENERATIONS}. */
    static void requireGenerationBudget(int prompts, int n) {
        if ((long) prompts * n > MAX_GENERATIONS) {
            throw ServerJson.badRequest(prompts + " prompts with n = " + n + " ask for more than "
                    + MAX_GENERATIONS + " generations in one request");
        }
    }

    /** Token ids in [0, vocabSize). */
    static int[] tokens(List<?> list, int vocabSize, String key) {
        int[] ids = new int[list.size()];
        for (int i = 0; i < ids.length; i++) {
            Object o = list.get(i);
            if (!(o instanceof Long || o instanceof Integer)) {
                throw ServerJson.badRequest(key + " must hold integer token ids");
            }
            long id = ((Number) o).longValue();
            if (id < 0 || id >= vocabSize) {
                throw ServerJson.badRequest(key + " token id " + id + " is outside the vocabulary [0, " + vocabSize
                        + ")");
            }
            ids[i] = (int) id;
        }
        return ids;
    }

    // ------------------------------------------------------------------ chat

    /**
     * {@link #chat(JsonView, boolean)} under the server's settings: request-supplied templates are trusted as
     * {@link ServerOptions#trustRequestChatTemplate} says, and the request renders over the default template
     * variables ({@link ServerOptions#chatTemplateKwargs}, {@link ChatRequest#withDefaultTemplateKwargs}).
     */
    static ChatRequest chat(JsonView body, ServerOptions options) {
        return chat(body, options.trustRequestChatTemplate()).withDefaultTemplateKwargs(options.chatTemplateKwargs());
    }

    /**
     * The chat request of the body: the messages, the template knobs (override and kwargs), the documents and the
     * tools with their choice.
     *
     * @param trustTemplate whether the server accepts a request-supplied template
     *                      ({@link ServerOptions#trustRequestChatTemplate}); without it a non-empty
     *                      {@code chat_template}, or a {@code chat_template} inside {@code chat_template_kwargs},
     *                      is refused with 400 as vLLM refuses it
     */
    static ChatRequest chat(JsonView body, boolean trustTemplate) {
        if (!(body.raw("messages") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("messages is required and must be a non-empty array");
        }
        ChatRequest.Builder chat = ChatRequest.builder();
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> message)) {
                throw ServerJson.badRequest("every message must be an object with role and content");
            }
            chat.addMessage(stringKeys(message));
        }
        if (body.has("add_generation_prompt")) {
            chat.addGenerationPrompt(body.optBool("add_generation_prompt", true));
        }
        if (body.has("continue_final_message")) {
            chat.continueFinalMessage(body.optBool("continue_final_message", false));
        }
        if (body.has("enable_thinking")) {
            chat.enableThinking(body.optBool("enable_thinking", false));
        }
        if (body.has("reasoning_effort")) {
            chat.reasoningEffort(body.str("reasoning_effort"));
        }
        if (body.raw("chat_template_kwargs") instanceof Map<?, ?> kwargs) {
            if (kwargs.get("chat_template") != null) {
                requireTrustedTemplate(trustTemplate, "chat_template_kwargs.chat_template");
            }
            for (Map.Entry<?, ?> e : kwargs.entrySet()) {
                chat.templateKwarg(String.valueOf(e.getKey()), e.getValue());
            }
        }
        if (body.has("chat_template")) {
            String template = body.str("chat_template");
            if (!template.isEmpty()) {
                // vLLM treats a falsy chat_template as unset
                requireTrustedTemplate(trustTemplate, "chat_template");
                chat.chatTemplate(template);
            }
        }
        refuseRequestMediaIo(body);
        chat.documents(documents(body));
        List<ChatTool> tools = tools(body);
        chat.tools(tools);
        ChatRequest.ToolChoice choice = toolChoice(body);
        if (choice != null) {
            if (tools.isEmpty() && choice.kind() != ChatRequest.ToolChoice.Kind.NONE
                    && choice.kind() != ChatRequest.ToolChoice.Kind.AUTO) {
                throw ServerJson.badRequest("tool_choice " + choice + " needs tools");
            }
            chat.toolChoice(choice);
        }
        try {
            return chat.build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /**
     * Refuses a request-supplied template on a server that does not trust them (vLLM's
     * {@code _validate_chat_template}), naming the {@code field} that carried it and the launch flag.
     */
    private static void requireTrustedTemplate(boolean trusted, String field) {
        if (!trusted) {
            throw ServerJson.badRequest(field + " is passed with the request, but the server runs without "
                    + ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_FLAG + " ("
                    + ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY + "); refused the untrusted chat template");
        }
    }

    /**
     * Refuses per-request media loader options: vLLM rejects a non-empty {@code media_io_kwargs} unless the server
     * trusts request multimodal kwargs, which this server does not offer; the loader options are the server's
     * {@code --media-io-kwargs}. A null or empty value is accepted, as in vLLM.
     */
    private static void refuseRequestMediaIo(JsonView body) {
        Object raw = body.raw("media_io_kwargs");
        if (raw == null || (raw instanceof Map<?, ?> m && m.isEmpty())) {
            return;
        }
        throw ServerJson.badRequest("media_io_kwargs: per-request media loader options are not accepted, as they"
                + " change the server's media loading; set them for the server with --media-io-kwargs");
    }

    /**
     * The request's {@code parallel_tool_calls}, or null when it does not set one (a JSON null counts as unset,
     * as llama.cpp's {@code json_value} reads it): the chat template's default then serves
     * ({@link EngineService#defaultParallelToolCalls}), and no guarantee is asked for.
     */
    static Boolean parallelToolCalls(JsonView body) {
        return body.raw("parallel_tool_calls") == null ? null : body.bool("parallel_tool_calls");
    }

    /** vLLM's {@code documents}: each an object a retrieval template renders into its context. */
    private static List<Map<String, Object>> documents(JsonView body) {
        Object raw = body.raw("documents");
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("documents must be an array of objects");
        }
        List<Map<String, Object>> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Map<?, ?> document)) {
                throw ServerJson.badRequest("documents[" + i + "] must be an object");
            }
            out.add(stringKeys(document));
        }
        return out;
    }

    private static List<ChatTool> tools(JsonView body) {
        List<ChatTool> out = new ArrayList<>();
        Object raw = body.raw("tools");
        if (raw == null) {
            return out;
        }
        if (!(raw instanceof List<?> list)) {
            throw ServerJson.badRequest("tools must be an array");
        }
        for (int i = 0; i < list.size(); i++) {
            if (!(list.get(i) instanceof Map<?, ?> tool) || !"function".equals(tool.get("type"))
                    || !(tool.get("function") instanceof Map<?, ?> fn)) {
                throw ServerJson.badRequest("tools[" + i + "] must be {\"type\": \"function\", \"function\": {...}}");
            }
            out.add(tool(stringKeys(fn), "tools[" + i + "].function", "parameters"));
        }
        return out;
    }

    /** A function declaration: name, optional description, optional parameter schema under {@code schemaKey}. */
    static ChatTool tool(Map<String, Object> fn, String where, String schemaKey) {
        if (!(fn.get("name") instanceof String name) || name.isEmpty()) {
            throw ServerJson.badRequest(where + ".name must be a non-empty string");
        }
        ChatTool.Builder b = ChatTool.builder(name);
        if (fn.get("description") instanceof String d) {
            b.description(d);
        }
        Object params = fn.get(schemaKey);
        if (params instanceof Map<?, ?> p) {
            b.parameters(stringKeys(p));
        } else if (params != null) {
            throw ServerJson.badRequest(where + "." + schemaKey + " must be a JSON schema object");
        }
        Object strict = fn.get("strict");
        if (strict instanceof Boolean flag) {
            b.strict(flag);
        } else if (strict != null) {
            throw ServerJson.badRequest(where + ".strict must be a boolean");
        }
        try {
            return b.build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(where + ": " + e.getMessage());
        }
    }

    private static ChatRequest.ToolChoice toolChoice(JsonView body) {
        Object raw = body.raw("tool_choice");
        if (raw == null) {
            return null;
        }
        if (raw instanceof String s) {
            try {
                return ChatRequest.ToolChoice.fromOpenAi(s);
            } catch (IllegalArgumentException e) {
                throw ServerJson.badRequest(e.getMessage());
            }
        }
        if (raw instanceof Map<?, ?> m && m.get("function") instanceof Map<?, ?> fn
                && fn.get("name") instanceof String name && !name.isEmpty()) {
            return ChatRequest.ToolChoice.function(name);
        }
        throw ServerJson.badRequest("tool_choice must be none, auto, required or"
                + " {\"type\": \"function\", \"function\": {\"name\": ...}}");
    }

    /** A JSON object with its keys as strings. */
    static Map<String, Object> stringKeys(Map<?, ?> map) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }
}
