package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The OpenAI response shapes shared by completions and chat completions: response and chunk heads, usage, finish
 * reasons and the two logprobs forms — chat's {@code {"content": [{token, logprob, bytes, top_logprobs}]}} and the
 * completions' legacy {@code {tokens, token_logprobs, top_logprobs, text_offset}}. A capture holds the sampled
 * token first and the most likely other tokens after it, so chat's {@code top} most likely tokens are the best of
 * the whole capture, and a completion's {@code top_logprobs} is vLLM's dict: the sampled token, then the
 * {@code top} most likely tokens, so up to {@code top + 1} entries as OpenAI documents. A capture of requested
 * {@code logprob_token_ids} lists the sampled token and every requested id, in order. An echoed completion's
 * arrays lead with the prompt tokens and their prompt logprobs, the first prompt token at null (vLLM's echo).
 * Tokens render through the model's tokenizer; an impossible token's logprob is -9999.0, as OpenAI reports it.
 */
final class OpenAiBodies {

    /** The logprob OpenAI reports for a token of zero probability. */
    static final double FLOOR_LOGPROB = -9999.0;

    private final EngineService service;
    private final int top;
    private final String model;
    /** Whether logprob tokens render as {@code token_id:{id}} ({@code return_tokens_as_token_ids}). */
    private final boolean asTokenIds;
    /** Whether captures hold requested {@code logprob_token_ids} rather than the best alternatives. */
    private final boolean explicitIds;

    /** Bodies naming {@code model} and listing {@code top} alternatives per token (-1 without logprobs). */
    OpenAiBodies(EngineService service, int top, String model) {
        this(service, top, model, false, false);
    }

    /** {@link #OpenAiBodies(EngineService, int, String)} with {@code return_tokens_as_token_ids}. */
    OpenAiBodies(EngineService service, int top, String model, boolean asTokenIds) {
        this(service, top, model, asTokenIds, false);
    }

    /**
     * {@link #OpenAiBodies(EngineService, int, String)} with {@code return_tokens_as_token_ids}, and with
     * {@code explicitIds} when the captures hold the request's {@code logprob_token_ids}.
     */
    OpenAiBodies(EngineService service, int top, String model, boolean asTokenIds, boolean explicitIds) {
        this.service = service;
        this.top = top;
        this.model = model;
        this.asTokenIds = asTokenIds;
        this.explicitIds = explicitIds;
    }

    /** The model name every response head carries. */
    String model() {
        return model;
    }

    /** A response or chunk head: id, object, created, model. */
    Map<String, Object> head(String id, String object, long created) {
        Map<String, Object> head = new LinkedHashMap<>();
        head.put("id", id);
        head.put("object", object);
        head.put("created", created);
        head.put("model", model);
        return head;
    }

    /** Usage over {@code promptTokens} (each distinct prompt counted once) and the choices' generated tokens. */
    static Map<String, Object> usage(long promptTokens, List<GenerationResult> results) {
        return usage(promptTokens, 0, results);
    }

    /**
     * Usage over the prompt tokens ({@code cachedTokens} of them served from the prefix cache, reported as
     * {@code prompt_tokens_details.cached_tokens}) and the choices' generated tokens.
     */
    static Map<String, Object> usage(long promptTokens, long cachedTokens, List<GenerationResult> results) {
        long completion = 0;
        for (GenerationResult r : results) {
            completion += r.timings().generatedTokens();
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", completion);
        usage.put("total_tokens", promptTokens + completion);
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("cached_tokens", cachedTokens);
        usage.put("prompt_tokens_details", details);
        return usage;
    }

    /**
     * {@code length} for the token budget or a full context, {@code abort} for a cancelled request,
     * {@code repetition} for a detected repeated pattern (vLLM's {@code FinishReason.REPETITION}), else {@code stop}.
     */
    static String finishReason(StopReason reason) {
        return switch (reason) {
            case MAX_TOKENS, CONTEXT_FULL -> "length";
            case CANCELLED -> "abort";
            case REPETITION -> "repetition";
            default -> "stop";
        };
    }

    // ------------------------------------------------------------------ chat form

    /** Chat's {@code logprobs} object over every capture. */
    Map<String, Object> chatLogprobs(List<TokenScore[]> steps) {
        List<Object> content = new ArrayList<>();
        for (TokenScore[] step : steps) {
            content.add(chatEntry(step));
        }
        return Map.of("content", content);
    }

    /** Chat's {@code logprobs} object of one streamed token. */
    Map<String, Object> chatLogprobs(TokenScore[] step) {
        return Map.of("content", List.of(chatEntry(step)));
    }

    private Map<String, Object> chatEntry(TokenScore[] step) {
        Map<String, Object> entry = tokenEntry(step[0]);
        List<Object> alternatives = new ArrayList<>();
        for (TokenScore t : explicitIds ? distinct(step) : ranked(step)) {
            alternatives.add(tokenEntry(t));
        }
        entry.put("top_logprobs", alternatives);
        return entry;
    }

    private Map<String, Object> tokenEntry(TokenScore t) {
        String text = asTokenIds ? "token_id:" + t.tokenId() : service.tokenText(t.tokenId());
        List<Object> bytes = new ArrayList<>();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            bytes.add(b & 0xff);
        }
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("token", text);
        entry.put("logprob", logprob(t));
        entry.put("bytes", bytes);
        return entry;
    }

    /**
     * vLLM's {@code prompt_logprobs} over every capture: null for the first prompt token (no row predicts it),
     * then one {@code {token_id: entry}} per prompt token, the predicted token leading with its vocabulary rank.
     */
    List<Object> promptLogprobs(List<TokenScore[]> captures) {
        List<Object> out = new ArrayList<>(captures.size());
        for (TokenScore[] capture : captures) {
            out.add(capture == null ? null : promptEntry(capture));
        }
        return out;
    }

    private Map<String, Object> promptEntry(TokenScore[] capture) {
        Map<String, Object> tokens = new LinkedHashMap<>();
        for (TokenScore t : capture) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("logprob", logprob(t));
            entry.put("rank", Math.max(1, t.rank()));
            entry.put("decoded_token", service.tokenText(t.tokenId()));
            tokens.put(String.valueOf(t.tokenId()), entry);
        }
        return tokens;
    }

    // ------------------------------------------------------------------ completions (legacy) form

    /**
     * The completions' {@code logprobs} object over every capture; {@code offset} is the character offset of the
     * first token in the choice text (the echoed prompt's length).
     */
    Map<String, Object> completionLogprobs(List<TokenScore[]> steps, int offset) {
        List<Object> tokens = new ArrayList<>();
        List<Object> tokenLogprobs = new ArrayList<>();
        List<Object> tops = new ArrayList<>();
        List<Object> offsets = new ArrayList<>();
        int at = offset;
        for (TokenScore[] step : steps) {
            String text = tokenText(step[0]);
            tokens.add(text);
            tokenLogprobs.add(logprob(step[0]));
            tops.add(topMap(step));
            offsets.add(at);
            at += text.length();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", tokens);
        out.put("token_logprobs", tokenLogprobs);
        out.put("top_logprobs", tops);
        out.put("text_offset", offsets);
        return out;
    }

    /**
     * The completions' {@code logprobs} object of an echoed choice ({@code echo} with {@code logprobs}): the
     * prompt's entries merged before the generated ones, as vLLM's serving_completion builds the arrays. Each
     * prompt position carries its prompt-logprobs capture, the prompt token leading its {@code top_logprobs}
     * ({@link #promptTopMap}); the first prompt token carries no row, so its token logprob and alternatives are
     * null. The offsets run from 0 over the token texts.
     */
    Map<String, Object> echoCompletionLogprobs(List<TokenScore[]> promptSteps, int[] promptTokens,
                                               List<TokenScore[]> steps) {
        List<Object> tokens = new ArrayList<>();
        List<Object> tokenLogprobs = new ArrayList<>();
        List<Object> tops = new ArrayList<>();
        List<Object> offsets = new ArrayList<>();
        int at = 0;
        for (int i = 0; i < promptTokens.length; i++) {
            TokenScore[] capture = i < promptSteps.size() ? promptSteps.get(i) : null;
            String text = tokenText(promptTokens[i]);
            tokens.add(text);
            tokenLogprobs.add(capture == null ? null : logprob(capture[0]));
            tops.add(capture == null ? null : promptTopMap(capture));
            offsets.add(at);
            at += text.length();
        }
        for (TokenScore[] step : steps) {
            String text = tokenText(step[0]);
            tokens.add(text);
            tokenLogprobs.add(logprob(step[0]));
            tops.add(topMap(step));
            offsets.add(at);
            at += text.length();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", tokens);
        out.put("token_logprobs", tokenLogprobs);
        out.put("top_logprobs", tops);
        out.put("text_offset", offsets);
        return out;
    }

    /** The character offset the generated tokens' entries take once the echoed prompt's entries merged before
     *  them: the prompt token texts' total length. */
    int echoLogprobsOffset(int[] promptTokens) {
        int at = 0;
        for (int token : promptTokens) {
            at += tokenText(token).length();
        }
        return at;
    }

    /** The completions' {@code logprobs} object of one streamed token at character {@code offset}. */
    Map<String, Object> completionLogprobs(TokenScore[] step, int offset) {
        return completionLogprobs(List.<TokenScore[]>of(step), offset);
    }

    /**
     * The {@code stop_reason} of a finished choice as vLLM reports it: the matched stop string, else the stop
     * token id the generation ended on, else null (EOS, length and every other reason).
     */
    static Object stopReason(GenerationResult result) {
        if (result.stopString() != null) {
            return result.stopString();
        }
        return result.stopTokenId() >= 0 ? result.stopTokenId() : null;
    }

    /**
     * A generated step's {@code top_logprobs} (vLLM's dict in insertion order): the sampled token, then the
     * {@link #top} most likely tokens; every requested id under {@code logprob_token_ids}.
     */
    private Map<String, Object> topMap(TokenScore[] step) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (explicitIds) {
            for (TokenScore t : distinct(step)) {
                map.putIfAbsent(tokenText(t), logprob(t));
            }
            return map;
        }
        map.put(tokenText(step[0]), logprob(step[0]));
        for (TokenScore t : ranked(step)) {
            map.putIfAbsent(tokenText(t), logprob(t));
        }
        return map;
    }

    /** A prompt position's {@code top_logprobs}: the first {@code top + 1} entries of its capture, in order. */
    private Map<String, Object> promptTopMap(TokenScore[] capture) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < capture.length && i <= Math.max(top, 0); i++) {
            map.putIfAbsent(tokenText(capture[i]), logprob(capture[i]));
        }
        return map;
    }

    /** The capture's entries in order, each token id once (vLLM's dict keyed by token id). */
    private static List<TokenScore> distinct(TokenScore[] step) {
        List<TokenScore> out = new ArrayList<>(step.length);
        Set<Integer> seen = new HashSet<>();
        for (TokenScore t : step) {
            if (seen.add(t.tokenId())) {
                out.add(t);
            }
        }
        return out;
    }

    /** A token's rendering in a logprobs body: its text, or {@code token_id:{id}} under {@code asTokenIds}. */
    private String tokenText(TokenScore t) {
        return tokenText(t.tokenId());
    }

    /** A token id's rendering: its text, or {@code token_id:{id}} under {@code asTokenIds}. */
    private String tokenText(int id) {
        return asTokenIds ? "token_id:" + id : service.tokenText(id);
    }

    /** The {@link #top} most likely tokens of a capture, most likely first. */
    private List<TokenScore> ranked(TokenScore[] step) {
        TokenScore[] sorted = step.clone();
        Arrays.sort(sorted, Comparator.comparingDouble((TokenScore t) -> t.logprob()).reversed());
        return Arrays.asList(sorted).subList(0, Math.max(0, Math.min(top, sorted.length)));
    }

    private static double logprob(TokenScore t) {
        double lp = t.logprob();
        return Double.isFinite(lp) ? lp : FLOOR_LOGPROB;
    }
}
