package com.eonmux.neuralfree.gguf.media;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The most media items of each modality one prompt may carry, vLLM's {@code --limit-mm-per-prompt}
 * (vllm/config/multimodal.py {@code limit_per_prompt}): a modality without an entry allows 999 items, and a limit
 * of 0 turns the modality off. Immutable.
 */
public final class MediaLimits {

    /** vLLM's limit of a modality the configuration leaves unset. */
    public static final int DEFAULT_LIMIT = 999;

    /** The media modalities a prompt counts. */
    public enum Modality {
        IMAGE, AUDIO, VIDEO;

        /** The lower-case name vLLM's configuration keys use. */
        public String key() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    private final Map<Modality, Integer> limits;

    private MediaLimits(Map<Modality, Integer> limits) {
        this.limits = limits;
    }

    /** Every modality at {@link #DEFAULT_LIMIT}. */
    public static MediaLimits defaults() {
        return new MediaLimits(new EnumMap<>(Modality.class));
    }

    /**
     * The limits of a JSON object such as {@code {"image": 2, "audio": {"count": 1}}} (vLLM's count-only and
     * object forms); null or blank is {@link #defaults()}.
     *
     * @throws IllegalArgumentException for text that is not a JSON object, a key that is not image, audio or
     *                                  video, an object form with a key other than {@code count}, or a count that
     *                                  is not a non-negative whole number
     */
    public static MediaLimits parse(String json) {
        if (json == null || json.trim().isEmpty()) {
            return defaults();
        }
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException("--limit-mm-per-prompt is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("--limit-mm-per-prompt must be a JSON object such as"
                    + " {\"image\": 2, \"audio\": 1}");
        }
        Map<Modality, Integer> out = new EnumMap<>(Modality.class);
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            Modality modality = modality(String.valueOf(e.getKey()));
            out.put(modality, count(modality, e.getValue()));
        }
        return new MediaLimits(out);
    }

    private static Modality modality(String key) {
        for (Modality m : Modality.values()) {
            if (m.key().equals(key)) {
                return m;
            }
        }
        throw new IllegalArgumentException("--limit-mm-per-prompt names the modality '" + key
                + "'; the modalities are image, audio and video");
    }

    private static int count(Modality modality, Object value) {
        Object raw = value;
        if (value instanceof Map) {
            Map<?, ?> options = (Map<?, ?>) value;
            for (Object k : options.keySet()) {
                if (!"count".equals(k)) {
                    throw new IllegalArgumentException("--limit-mm-per-prompt " + modality.key() + " option '" + k
                            + "' is not supported; only count is");
                }
            }
            raw = options.containsKey("count") ? options.get("count") : DEFAULT_LIMIT;
        }
        if (!(raw instanceof Number) || ((Number) raw).doubleValue() != Math.rint(((Number) raw).doubleValue())
                || ((Number) raw).longValue() < 0 || ((Number) raw).longValue() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("--limit-mm-per-prompt " + modality.key()
                    + " must be a non-negative whole number, got " + raw);
        }
        return ((Number) raw).intValue();
    }

    /** The most items of {@code modality} one prompt may carry. */
    public int limit(Modality modality) {
        Integer v = limits.get(Objects.requireNonNull(modality, "modality"));
        return v == null ? DEFAULT_LIMIT : v;
    }

    /**
     * Refuses a prompt carrying more than the limit of {@code modality}, with vLLM's message.
     *
     * @throws MediaInputException when {@code count} exceeds the limit
     */
    public void check(Modality modality, int count) {
        int limit = limit(modality);
        if (count > limit) {
            throw new MediaInputException("At most " + limit + " " + modality.key() + "(s) may be provided in"
                    + " one prompt.");
        }
    }

    @Override
    public String toString() {
        return "MediaLimits{image=" + limit(Modality.IMAGE) + ", audio=" + limit(Modality.AUDIO) + ", video="
                + limit(Modality.VIDEO) + '}';
    }
}
