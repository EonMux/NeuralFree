package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The OpenAI response shapes shared by completions and chat completions: response and chunk heads, usage, finish
 * reasons and the two logprobs forms — chat's {@code {"content": [{token, logprob, bytes, top_logprobs}]}} and the
 * completions' legacy {@code {tokens, token_logprobs, top_logprobs, text_offset}}. A capture holds the sampled
 * token first and the most likely other tokens after it, so the {@code top} most likely tokens are the best of the
 * whole capture. Tokens render through the model's tokenizer; an impossible token's logprob is -9999.0, as
 * OpenAI reports it.
 */
final class OpenAiBodies {

    /** The logprob OpenAI reports for a token of zero probability. */
    static final double FLOOR_LOGPROB = -9999.0;

    private final EngineService service;
    private final int top;
    private final String model;

    /** Bodies naming {@code model} and listing {@code top} alternatives per token (-1 without logprobs). */
    OpenAiBodies(EngineService service, int top, String model) {
        this.service = service;
        this.top = top;
        this.model = model;
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
        long completion = 0;
        for (GenerationResult r : results) {
            completion += r.timings().generatedTokens();
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", completion);
        usage.put("total_tokens", promptTokens + completion);
        return usage;
    }

    /** {@code length} for the token budget or a full context, else {@code stop}. */
    static String finishReason(StopReason reason) {
        return switch (reason) {
            case MAX_TOKENS, CONTEXT_FULL -> "length";
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
        for (TokenScore t : ranked(step)) {
            alternatives.add(tokenEntry(t));
        }
        entry.put("top_logprobs", alternatives);
        return entry;
    }

    private Map<String, Object> tokenEntry(TokenScore t) {
        String text = service.tokenText(t.tokenId());
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
            String text = service.tokenText(step[0].tokenId());
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

    /** The completions' {@code logprobs} object of one streamed token at character {@code offset}. */
    Map<String, Object> completionLogprobs(TokenScore[] step, int offset) {
        return completionLogprobs(List.<TokenScore[]>of(step), offset);
    }

    private Map<String, Object> topMap(TokenScore[] step) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (TokenScore t : ranked(step)) {
            map.putIfAbsent(service.tokenText(t.tokenId()), logprob(t));
        }
        return map;
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
