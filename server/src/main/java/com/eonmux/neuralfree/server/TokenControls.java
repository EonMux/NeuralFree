package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.RepetitionDetection;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The vLLM per-request token controls of a request body onto the engine request: {@code stop_token_ids},
 * {@code allowed_token_ids}, {@code include_stop_str_in_output}, {@code skip_special_tokens},
 * {@code spaces_between_special_tokens} and {@code repetition_detection}. A null value means the field's default,
 * as in vLLM's request models; a malformed value refuses with 400 naming the field.
 */
final class TokenControls {

    /** The field of the n-gram repetition stop. */
    static final String REPETITION_DETECTION = "repetition_detection";

    /** The members of a {@code repetition_detection} object (vLLM's {@code RepetitionDetectionParams}). */
    private static final Set<String> REPETITION_MEMBERS = Set.of("max_pattern_size", "min_pattern_size", "min_count");

    private TokenControls() {
    }

    /** Applies the body's token controls to {@code builder}; token ids must lie in {@code [0, vocabSize)}. */
    static void apply(JsonView body, GenerationRequest.Builder builder, int vocabSize) {
        Object stops = body.raw("stop_token_ids");
        if (stops != null) {
            builder.stopTokenIds(ids(stops, "stop_token_ids", vocabSize, true));
        }
        Object allowed = body.raw("allowed_token_ids");
        if (allowed != null) {
            builder.allowedTokenIds(ids(allowed, "allowed_token_ids", vocabSize, false));
        }
        builder.includeStopString(flag(body, "include_stop_str_in_output", false));
        builder.skipSpecialTokens(flag(body, "skip_special_tokens", true));
        builder.spacesBetweenSpecialTokens(flag(body, "spaces_between_special_tokens", true));
        builder.repetitionDetection(repetitionDetection(body.raw(REPETITION_DETECTION)));
    }

    /**
     * vLLM's {@code repetition_detection} object as the engine's detection: {@code max_pattern_size},
     * {@code min_pattern_size} and {@code min_count}, each an integer defaulting to 0 when absent or null; null
     * disables detection. Validated as vLLM's {@code RepetitionDetectionParams}.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming {@code repetition_detection} for a value that is not
     *                                               such an object, an unknown member, a non-integer member or an
     *                                               invalid combination
     */
    static RepetitionDetection repetitionDetection(Object raw) {
        if (raw == null) {
            return RepetitionDetection.disabled();
        }
        if (!(raw instanceof Map<?, ?> members)) {
            throw ServerJson.badRequest(REPETITION_DETECTION + " must be an object with max_pattern_size,"
                    + " min_pattern_size and min_count");
        }
        for (Object key : members.keySet()) {
            if (!REPETITION_MEMBERS.contains(key)) {
                throw ServerJson.badRequest(REPETITION_DETECTION + " has no member '" + key
                        + "'; it takes max_pattern_size, min_pattern_size and min_count");
            }
        }
        int max = member(members, "max_pattern_size");
        int min = member(members, "min_pattern_size");
        int count = member(members, "min_count");
        try {
            return RepetitionDetection.of(max, min, count);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** An integer member, 0 when absent or null; values beyond the {@code int} range saturate. */
    private static int member(Map<?, ?> members, String key) {
        Object v = members.get(key);
        if (v == null) {
            return 0;
        }
        long value;
        if (v instanceof Integer || v instanceof Long) {
            value = ((Number) v).longValue();
        } else if (v instanceof BigInteger big) {
            value = big.signum() < 0 ? Long.MIN_VALUE : Long.MAX_VALUE;
        } else {
            throw ServerJson.badRequest(REPETITION_DETECTION + "." + key + " must be an integer, got " + v);
        }
        return (int) Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, value));
    }

    /**
     * Integer token ids in {@code [0, vocabSize)}: an array, which only {@code emptyAllowed} fields may leave empty
     * (vLLM's {@code stop_token_ids} defaults to {@code []}; an empty {@code allowed_token_ids} refuses).
     */
    static int[] ids(Object raw, String key, int vocabSize, boolean emptyAllowed) {
        if (!(raw instanceof List<?> list) || (list.isEmpty() && !emptyAllowed)) {
            throw ServerJson.badRequest(key + (emptyAllowed ? " must be an array of token ids"
                    : " must be a non-empty array of token ids"));
        }
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            Object o = list.get(i);
            if (!(o instanceof Long || o instanceof Integer)) {
                throw ServerJson.badRequest(key + " must hold integer token ids");
            }
            long id = ((Number) o).longValue();
            if (id < 0 || id >= vocabSize) {
                throw ServerJson.badRequest(key + " token id " + id + " is outside the vocabulary [0, "
                        + vocabSize + ")");
            }
            out[i] = (int) id;
        }
        return out;
    }

    /** A boolean field, {@code dflt} when absent or null. */
    private static boolean flag(JsonView body, String key, boolean dflt) {
        Object v = body.raw(key);
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof Boolean b)) {
            throw ServerJson.badRequest(key + " must be a boolean, got " + v);
        }
        return b;
    }
}
