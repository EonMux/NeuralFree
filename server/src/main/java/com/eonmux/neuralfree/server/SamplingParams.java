package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * The sampler settings of a request body: the OpenAI fields, vLLM's {@code repetition_penalty} and {@code min_p},
 * and llama.cpp's sampling fields ({@code tools/server/server-schema.cpp}): {@code dynatemp_range},
 * {@code dynatemp_exponent}, {@code top_n_sigma}, {@code xtc_probability}, {@code xtc_threshold},
 * {@code repeat_last_n}, the {@code dry_*} family, {@code mirostat}, {@code mirostat_tau}, {@code mirostat_eta},
 * {@code adaptive_target}, {@code adaptive_decay}, {@code min_keep} and the {@code samplers} order (an array of stage
 * names or a string of stage characters). Absent fields keep the engine defaults; malformed or out-of-range values
 * refuse with 400 ({@link SamplerConfig.Builder#build} checks the ranges); a JSON null keeps the default.
 */
final class SamplingParams {

    /** llama.cpp's {@code LLAMA_DRY_MAX_SEQ_BREAKERS}: the distinct DRY sequence breakers one request may name. */
    static final int MAX_DRY_SEQUENCE_BREAKERS = 64;

    private SamplingParams() {
    }

    /** The sampler of {@code body} over the engine defaults. */
    static SamplerConfig of(JsonView body) {
        SamplerConfig.Builder s = GenerationRequest.defaults().sampler().toBuilder();
        try {
            basic(body, s);
            llamaCpp(body, s);
            if (present(body, "seed")) {
                long seed = body.i64("seed");
                if (seed >= 0) {
                    s.seed(seed);
                }
            }
            return s.build();
        } catch (IllegalArgumentException | JsonException e) {
            throw ServerJson.badRequest("invalid sampling parameters: " + e.getMessage());
        }
    }

    /** The OpenAI and vLLM fields. */
    private static void basic(JsonView body, SamplerConfig.Builder s) {
        real(body, "temperature", s::temperature);
        real(body, "top_p", s::topP);
        whole(body, "top_k", s::topK);
        real(body, "min_p", s::minP);
        real(body, "typical_p", s::typicalP);
        real(body, body.has("repeat_penalty") ? "repeat_penalty" : "repetition_penalty", s::repeatPenalty);
        real(body, "presence_penalty", s::presencePenalty);
        real(body, "frequency_penalty", s::frequencyPenalty);
        whole(body, "repeat_last_n", s::penaltyLastN);
        whole(body, "min_keep", s::minKeep);
    }

    /** llama.cpp's extended sampler fields. */
    private static void llamaCpp(JsonView body, SamplerConfig.Builder s) {
        real(body, "dynatemp_range", s::dynatempRange);
        real(body, "dynatemp_exponent", s::dynatempExponent);
        real(body, "top_n_sigma", s::topNSigma);
        real(body, "xtc_probability", s::xtcProbability);
        real(body, "xtc_threshold", s::xtcThreshold);
        real(body, "dry_multiplier", s::dryMultiplier);
        real(body, "dry_base", s::dryBase);
        whole(body, "dry_allowed_length", s::dryAllowedLength);
        whole(body, "dry_penalty_last_n", s::dryPenaltyLastN);
        whole(body, "mirostat", s::mirostat);
        real(body, "mirostat_tau", s::mirostatTau);
        real(body, "mirostat_eta", s::mirostatEta);
        real(body, "adaptive_target", s::adaptiveTarget);
        real(body, "adaptive_decay", s::adaptiveDecay);
        if (present(body, "dry_sequence_breakers")) {
            s.drySequenceBreakers(sequenceBreakers(body));
        }
        if (present(body, "samplers")) {
            s.samplers(order(body.raw("samplers")));
        }
    }

    /** llama.cpp's breaker rule: a non-empty array of strings, at most 64 distinct ones, duplicates dropped. */
    private static List<String> sequenceBreakers(JsonView body) {
        if (!(body.raw("dry_sequence_breakers") instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest("dry_sequence_breakers must be a non-empty array of strings");
        }
        Set<String> unique = new LinkedHashSet<>();
        for (Object o : list) {
            if (!(o instanceof String str)) {
                throw ServerJson.badRequest("dry_sequence_breakers must be a non-empty array of strings");
            }
            unique.add(str);
        }
        if (unique.size() > MAX_DRY_SEQUENCE_BREAKERS) {
            throw ServerJson.badRequest("dry_sequence_breakers must contain at most " + MAX_DRY_SEQUENCE_BREAKERS
                    + " distinct entries");
        }
        return new ArrayList<>(unique);
    }

    /** The stage order of a {@code samplers} array of names or string of characters. */
    private static List<StageKind> order(Object raw) {
        if (raw instanceof String chars) {
            return StageKind.ofChars(chars);
        }
        if (raw instanceof List<?> names) {
            List<StageKind> out = new ArrayList<>(names.size());
            for (Object o : names) {
                if (!(o instanceof String name)) {
                    throw ServerJson.badRequest("samplers must be an array of sampler names or a string of"
                            + " sampler characters");
                }
                out.add(StageKind.ofName(name));
            }
            return out;
        }
        throw ServerJson.badRequest("samplers must be an array of sampler names or a string of sampler characters");
    }

    /** Whether the body sets {@code key}; an explicit JSON null leaves the default, as OpenAI clients send it. */
    private static boolean present(JsonView body, String key) {
        return body.raw(key) != null;
    }

    private static void real(JsonView body, String key, Consumer<Float> setter) {
        if (present(body, key)) {
            setter.accept((float) body.f64(key));
        }
    }

    private static void whole(JsonView body, String key, Consumer<Integer> setter) {
        if (present(body, key)) {
            setter.accept(body.i32(key));
        }
    }
}
