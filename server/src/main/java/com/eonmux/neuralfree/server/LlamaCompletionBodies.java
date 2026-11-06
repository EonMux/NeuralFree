package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The bodies of llama.cpp's native completion ({@code server_task_result_cmpl_final/_partial::to_json_non_oaicompat},
 * {@code tools/server/server-task.cpp}): the final object with {@code content}, {@code tokens}, {@code stop_type},
 * {@code generation_settings} and {@code timings}, the streamed partial chunks, and {@code completion_probabilities}
 * when {@code n_probs} asks for them. The probabilities are the model's own ({@code logprob}, {@code top_logprobs})
 * unless {@code post_sampling_probs} asks for the sampler chain's ({@code prob}, {@code top_probs}, tokens the
 * chain removed left out). The final object lists the generated token ids under {@code tokens} only with
 * {@code return_tokens}; a streamed chunk always carries its token. {@code response_fields} selects nested fields
 * of the final object.
 */
final class LlamaCompletionBodies {

    private final EngineService service;
    private final String model;
    private final int nProbs;
    private final boolean postSampling;
    private final boolean returnTokens;
    private final List<String> fields;

    /**
     * @param nProbs       alternatives per token in {@code completion_probabilities}, 0 for none
     * @param postSampling whether the captures score the sampler chain's distribution
     *                     ({@code post_sampling_probs}), rendered as probabilities
     * @param returnTokens whether the final object lists the generated token ids ({@code return_tokens})
     * @param fields       the {@code response_fields} paths ({@code a/b}), empty for the whole object
     */
    LlamaCompletionBodies(EngineService service, String model, int nProbs, boolean postSampling,
                          boolean returnTokens, List<String> fields) {
        this.service = service;
        this.model = model;
        this.nProbs = nProbs;
        this.postSampling = postSampling;
        this.returnTokens = returnTokens;
        this.fields = List.copyOf(fields);
    }

    /**
     * The final object of choice {@code index}; under {@code streamed} its content and tokens are empty, since the
     * partial chunks carried them, and without {@code return_tokens} its tokens are empty too. A media prompt counts its media rows as evaluated and decodes its text tokens.
     */
    Map<String, Object> finalBody(int index, LlamaPrompts.Prompt prompt, GenerationRequest request,
                                  GenerationResult result,
                                  BatchSequence sequence, boolean streamed) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("index", index);
        out.put("content", streamed ? "" : result.text());
        out.put("tokens", streamed || !returnTokens ? List.of() : ids(result.tokens()));
        out.put("id_slot", sequence == null ? -1 : sequence.seq());
        out.put("stop", true);
        out.put("model", model);
        out.put("tokens_predicted", result.tokens().length);
        out.put("tokens_evaluated", prompt.rows());
        out.put("generation_settings", settings(request, streamed, nProbs, postSampling));
        out.put("prompt", service.decode(prompt.tokens()));
        out.put("has_new_line", result.text().indexOf('\n') >= 0);
        out.put("truncated", sequence != null && sequence.truncated());
        out.put("stop_type", stopType(result.stopReason()));
        out.put("stopping_word", result.stopString() == null ? "" : result.stopString());
        out.put("tokens_cached", prompt.rows() + result.tokens().length);
        out.put("timings", timings(result, sequence == null ? 0 : sequence.reusedPrefix()));
        if (!streamed && nProbs > 0 && !result.logprobs().isEmpty()) {
            List<Object> probs = new ArrayList<>();
            for (TokenScore[] step : result.logprobs()) {
                probs.add(probability(step));
            }
            out.put("completion_probabilities", probs);
        }
        return fields.isEmpty() ? out : select(out);
    }

    /** A streamed chunk of choice {@code index}: one token's released text and the token itself. */
    Map<String, Object> partial(int index, GenerationBatch.Text text, int evaluated, BatchSequence sequence) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("index", index);
        out.put("content", text.text());
        out.put("tokens", text.tokenId() < 0 ? List.of() : List.of(text.tokenId()));
        out.put("stop", false);
        out.put("id_slot", sequence == null ? -1 : sequence.seq());
        out.put("tokens_predicted", text.predicted());
        out.put("tokens_evaluated", evaluated);
        if (nProbs > 0 && text.logprobs() != null) {
            out.put("completion_probabilities", List.of(probability(text.logprobs())));
        }
        return out;
    }

    /**
     * llama.cpp's {@code stop_type}: eos, word (a stop string) or limit (the budget, the context, or a detected
     * repetition, as llama.cpp reports its own heuristic early stops).
     */
    static String stopType(StopReason reason) {
        switch (reason) {
            case EOS:
                return "eos";
            case STOP_STRING:
                return "word";
            case MAX_TOKENS:
            case CONTEXT_FULL:
            case REPETITION:
                return "limit";
            default:
                return "none";
        }
    }

    private static Map<String, Object> settings(GenerationRequest request, boolean streamed, int nProbs,
                                                boolean postSampling) {
        SamplerConfig s = request.sampler();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("seed", s.seed());
        out.put("temperature", s.temperature());
        out.put("dynatemp_range", s.dynatempRange());
        out.put("dynatemp_exponent", s.dynatempExponent());
        out.put("top_k", s.topK());
        out.put("top_p", s.topP());
        out.put("min_p", s.minP());
        out.put("top_n_sigma", s.topNSigma());
        out.put("xtc_probability", s.xtcProbability());
        out.put("xtc_threshold", s.xtcThreshold());
        out.put("typical_p", s.typicalP());
        out.put("repeat_last_n", s.penaltyLastN());
        out.put("repeat_penalty", s.repeatPenalty());
        out.put("presence_penalty", s.presencePenalty());
        out.put("frequency_penalty", s.frequencyPenalty());
        out.put("dry_multiplier", s.dryMultiplier());
        out.put("dry_base", s.dryBase());
        out.put("dry_allowed_length", s.dryAllowedLength());
        out.put("dry_penalty_last_n", s.dryPenaltyLastN());
        out.put("dry_sequence_breakers", s.drySequenceBreakers());
        out.put("mirostat", s.mirostat());
        out.put("mirostat_tau", s.mirostatTau());
        out.put("mirostat_eta", s.mirostatEta());
        out.put("adaptive_target", s.adaptiveTarget());
        out.put("adaptive_decay", s.adaptiveDecay());
        int budget = request.unlimited() ? -1 : request.maxNewTokens();
        out.put("max_tokens", budget);
        out.put("n_predict", budget);
        out.put("stop", request.stopStrings());
        out.put("ignore_eos", request.ignoreEos());
        out.put("stream", streamed);
        out.put("n_probs", nProbs);
        out.put("min_keep", s.minKeep());
        List<String> samplers = new ArrayList<>();
        for (StageKind kind : s.order()) {
            samplers.add(kind.label());
        }
        out.put("samplers", samplers);
        out.put("post_sampling_probs", postSampling);
        return out;
    }

    /** {@code server_slot_stats::to_json}: the prompt and generation counts, times and rates. */
    private static Map<String, Object> timings(GenerationResult result, int cached) {
        GenerationResult.Timings t = result.timings();
        double promptMs = t.promptNanos() / 1e6;
        double genMs = t.generationNanos() / 1e6;
        int promptN = Math.max(0, t.promptTokens() - cached);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("cache_n", cached);
        out.put("prompt_n", promptN);
        out.put("prompt_ms", promptMs);
        out.put("prompt_per_token_ms", promptN == 0 ? 0.0 : promptMs / promptN);
        out.put("prompt_per_second", promptMs <= 0 ? 0.0 : 1e3 * promptN / promptMs);
        out.put("predicted_n", t.generatedTokens());
        out.put("predicted_ms", genMs);
        out.put("predicted_per_token_ms", t.generatedTokens() == 0 ? 0.0 : genMs / t.generatedTokens());
        out.put("predicted_per_second", genMs <= 0 ? 0.0 : 1e3 * t.generatedTokens() / genMs);
        return out;
    }

    /**
     * One token's entry: the sampled token and its {@link #nProbs} most likely alternatives; post-sampling, the
     * alternatives stop at the first the chain removed (zero probability), as llama.cpp filters them.
     */
    private Map<String, Object> probability(TokenScore[] step) {
        Map<String, Object> entry = scoreEntry(step[0]);
        TokenScore[] ranked = step.clone();
        Arrays.sort(ranked, Comparator.comparingDouble((TokenScore t) -> t.logprob()).reversed());
        List<Object> top = new ArrayList<>();
        for (int i = 0; i < Math.min(nProbs, ranked.length); i++) {
            if (postSampling && Math.exp(ranked[i].logprob()) == 0.0) {
                break;
            }
            top.add(scoreEntry(ranked[i]));
        }
        entry.put(postSampling ? "top_probs" : "top_logprobs", top);
        return entry;
    }

    private Map<String, Object> scoreEntry(TokenScore t) {
        String text = service.tokenText(t.tokenId());
        List<Object> bytes = new ArrayList<>();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            bytes.add(b & 0xff);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", t.tokenId());
        out.put("token", text);
        out.put("bytes", bytes);
        double lp = t.logprob();
        if (postSampling) {
            out.put("prob", Math.exp(lp));
        } else {
            out.put("logprob", Double.isFinite(lp) ? lp : OpenAiBodies.FLOOR_LOGPROB);
        }
        return out;
    }

    /** {@code json_get_nested_values}: each {@code a/b} path that resolves, keyed by the path. */
    private Map<String, Object> select(Map<String, Object> body) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String path : fields) {
            Object current = body;
            boolean found = true;
            for (String key : path.split("/", -1)) {
                if (current instanceof Map<?, ?> map && map.containsKey(key)) {
                    current = map.get(key);
                } else {
                    found = false;
                    break;
                }
            }
            if (found) {
                out.put(path, current);
            }
        }
        return out;
    }

    private static List<Integer> ids(int[] tokens) {
        List<Integer> out = new ArrayList<>(tokens.length);
        for (int t : tokens) {
            out.add(t);
        }
        return out;
    }
}
