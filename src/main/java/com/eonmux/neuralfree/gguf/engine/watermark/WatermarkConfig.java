package com.eonmux.neuralfree.gguf.engine.watermark;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The settings of text watermarking, matching vLLM's {@code WatermarkConfig}: the secret 64-bit key, the algorithm
 * ({@link WatermarkAlgorithm}), the dual-key routing weight {@code alpha}, the number of prior tokens the
 * pseudorandom function keys on ({@code context_width}), the repeated-context search
 * ({@code deduplicate_contexts}, {@code deduplicate_contexts_max_history}) and the pseudorandom function
 * ({@link WatermarkPrfName}). Immutable.
 *
 * <p>The configuration is read as a JSON object from the {@value #PROP_CONFIG} property (vLLM's
 * {@code --watermark-config}) or with {@link #ofJson(Object)}. The keys are vLLM's fields: {@code key},
 * {@code algorithm}, {@code alpha}, {@code context_width}, {@code deduplicate_contexts},
 * {@code deduplicate_contexts_max_history}, {@code prf} and {@code allow_target_only_watermarking}. A key the
 * configuration does not name is refused by name.
 *
 * <p>The secret key is best kept out of the command line, where it would appear in
 * {@code /proc/<pid>/cmdline} and shell history. {@link #fromProperties(Properties)} therefore also reads it from
 * the {@value #ENV_KEY} environment variable or from the file the {@value #ENV_KEY_FILE} environment variable
 * names; the JSON {@code key} remains accepted for vLLM parity. {@value #ENV_KEY} takes precedence over
 * {@value #ENV_KEY_FILE}, and an environment source beside the JSON {@code key} is refused rather than silently
 * choosing one. The environment alone is enough to turn watermarking on at the default settings.
 *
 * <p>The key is a 64-bit unsigned value held as a {@code long} bit pattern; it is a secret, so {@link #toString},
 * the error messages and the log lines leave it out.
 */
public final class WatermarkConfig {

    /** The property carrying the configuration as a JSON object (vLLM's {@code --watermark-config}). */
    public static final String PROP_CONFIG = "neuralfree.watermark.config";

    /** The environment variable carrying the secret key as an unsigned 64-bit decimal integer. */
    public static final String ENV_KEY = "NEURALFREE_WATERMARK_KEY";

    /** The environment variable naming a file whose text is the secret key (trimmed). */
    public static final String ENV_KEY_FILE = "NEURALFREE_WATERMARK_KEY_FILE";

    /** Marks that the warning about a key on the command line has been logged. */
    private static final AtomicBoolean COMMAND_LINE_KEY_WARNED = new AtomicBoolean();

    private static final Logger LOGGER = LogManager.getLogger(WatermarkConfig.class);


    /** The default probability of routing a dual-key token to key B. */
    public static final double DEFAULT_ALPHA = 0.1;
    /** The default number of prior tokens the pseudorandom function keys on. */
    public static final int DEFAULT_CONTEXT_WIDTH = 4;
    /** The default number of history positions the repeated-context search scans. */
    public static final Integer DEFAULT_DEDUPLICATE_MAX_HISTORY = 8192;

    private static final Set<String> KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "key", "algorithm", "alpha", "context_width", "deduplicate_contexts",
            "deduplicate_contexts_max_history", "prf", "allow_target_only_watermarking")));

    private static final BigInteger MAX_UNSIGNED_64 = BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE);
    private static final String KEY_A_DOMAIN = "key_a";
    private static final String KEY_B_DOMAIN = "key_b";

    private final long key;
    private final WatermarkAlgorithm algorithm;
    private final double alpha;
    private final int contextWidth;
    private final WatermarkContextScope deduplicateContexts;
    private final Integer deduplicateContextsMaxHistory;
    private final WatermarkPrfName prf;
    private final boolean allowTargetOnlyWatermarking;

    private WatermarkConfig(Builder b) {
        this.key = b.key;
        this.algorithm = b.algorithm;
        this.alpha = b.alpha;
        this.contextWidth = b.contextWidth;
        this.deduplicateContexts = b.deduplicateContexts;
        this.deduplicateContextsMaxHistory = b.deduplicateContextsMaxHistory;
        this.prf = b.prf;
        this.allowTargetOnlyWatermarking = b.allowTargetOnlyWatermarking;
    }

    /** A configuration of the default algorithm and PRF under {@code key}. */
    public static WatermarkConfig of(long key) {
        return builder(key).build();
    }

    /** A builder whose key is set to {@code key}. */
    public static Builder builder(long key) {
        return new Builder().key(key);
    }

    /** A builder whose key has to be given before {@link Builder#build()}. */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * The configuration {@code p} carries in {@value #PROP_CONFIG}, or null when neither the property nor an
     * environment source configures a watermark. The key is the JSON {@code key} unless the environment supplies
     * it (see {@link #fromProperties(Properties, Map)}).
     *
     * @throws IllegalArgumentException when the property is not a JSON object, names a field this configuration
     *                                  does not have, has no key and no environment source supplies one, or an
     *                                  environment source and the JSON key are both given
     */
    public static WatermarkConfig fromProperties(Properties p) {
        return fromProperties(p, System.getenv());
    }

    /**
     * {@link #fromProperties(Properties)} over an explicit environment, so the sources are testable. The key may
     * come from the {@value #ENV_KEY} environment variable or from the file named by {@value #ENV_KEY_FILE},
     * which is trimmed and read as one unsigned 64-bit decimal; the variable wins over the file. The JSON
     * {@code key} is accepted for vLLM parity and may not be combined with an environment source.
     *
     * @throws IllegalArgumentException when the property is not a JSON object, names a field this configuration
     *                                  does not have, the key source is malformed, no source supplies a key, or
     *                                  more than one kind of source does
     */
    public static WatermarkConfig fromProperties(Properties p, Map<String, String> env) {
        Objects.requireNonNull(p, "properties");
        Objects.requireNonNull(env, "env");
        KeyFromEnvironment environmentKey = environmentKey(env);
        String configured = p.getProperty(PROP_CONFIG);
        if (configured == null || configured.trim().isEmpty()) {
            return environmentKey == null ? null : builder(environmentKey.value).build();
        }
        Object parsed;
        try {
            parsed = Json.parse(configured.trim(), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(PROP_CONFIG + " is not valid JSON at position " + e.position(), e);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(PROP_CONFIG + " is not valid JSON", e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(PROP_CONFIG + " must be a JSON object, got "
                    + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        Map<?, ?> map = (Map<?, ?>) parsed;
        boolean jsonKey = map.containsKey("key") && map.get("key") != null;
        if (environmentKey != null && jsonKey) {
            throw new IllegalArgumentException("the watermark key is given by both " + environmentKey.source
                    + " and the watermark config 'key'; keep one source");
        }
        if (environmentKey == null && !jsonKey) {
            throw new IllegalArgumentException("watermark config has no 'key'; give it in the config, " + ENV_KEY
                    + " or " + ENV_KEY_FILE);
        }
        if (environmentKey != null) {
            return build(map, environmentKey.value);
        }
        warnCommandLineKeyOnce();
        return build(map, parseKey(map.get("key")));
    }

    /**
     * The configuration of a parsed JSON object with vLLM's field names and its required {@code key}.
     *
     * @throws IllegalArgumentException when {@code parsed} is not an object, a field is unknown or a value is
     *                                  malformed or out of range, or the required {@code key} is absent
     */
    public static WatermarkConfig ofJson(Object parsed) {
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(PROP_CONFIG + " must be a JSON object, got "
                    + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        Map<?, ?> map = (Map<?, ?>) parsed;
        if (!map.containsKey("key") || map.get("key") == null) {
            throw new IllegalArgumentException("watermark config has no 'key'; give it in the config, " + ENV_KEY
                    + " or " + ENV_KEY_FILE);
        }
        return build(map, parseKey(map.get("key")));
    }

    /** The configuration of {@code map} under {@code key}, refusing an unknown field or a bad value by name. */
    private static WatermarkConfig build(Map<?, ?> map, long key) {
        Map<String, Object> fields = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            String name = String.valueOf(e.getKey());
            if (!KEYS.contains(name)) {
                throw new IllegalArgumentException("watermark config key '" + name + "' is not supported; the"
                        + " supported keys are " + String.join(", ", KEYS));
            }
            fields.put(name, e.getValue());
        }
        Builder b = builder(key);
        Object algorithm = fields.get("algorithm");
        if (algorithm != null) {
            b.algorithm(WatermarkAlgorithm.ofLabel(string(algorithm, "algorithm")));
        }
        Object alpha = fields.get("alpha");
        if (alpha != null) {
            b.alpha(number(alpha, "alpha"));
        }
        Object contextWidth = fields.get("context_width");
        if (contextWidth != null) {
            b.contextWidth(whole(contextWidth, "context_width"));
        }
        Object dedup = fields.get("deduplicate_contexts");
        if (dedup != null) {
            b.deduplicateContexts(WatermarkContextScope.ofLabel(string(dedup, "deduplicate_contexts")));
        }
        if (fields.containsKey("deduplicate_contexts_max_history")) {
            Object maxHistory = fields.get("deduplicate_contexts_max_history");
            b.deduplicateContextsMaxHistory(maxHistory == null ? null
                    : whole(maxHistory, "deduplicate_contexts_max_history"));
        }
        Object prf = fields.get("prf");
        if (prf != null) {
            b.prf(WatermarkPrfName.ofLabel(string(prf, "prf")));
        }
        Object allowTargetOnly = fields.get("allow_target_only_watermarking");
        if (allowTargetOnly != null) {
            if (!(allowTargetOnly instanceof Boolean)) {
                throw new IllegalArgumentException("watermark config allow_target_only_watermarking must be a"
                        + " boolean, got " + describe(allowTargetOnly));
            }
            b.allowTargetOnlyWatermarking((Boolean) allowTargetOnly);
        }
        return b.build();
    }

    /** The key an environment source supplies: its value and the variable that supplied it. */
    private static final class KeyFromEnvironment {

        private final long value;
        private final String source;

        private KeyFromEnvironment(long value, String source) {
            this.value = value;
            this.source = source;
        }
    }

    /**
     * The key of {@code env}: {@value #ENV_KEY} first, else the file {@value #ENV_KEY_FILE} names. A variable
     * that is set but blank is refused, as is malformed content; neither message repeats the value.
     */
    private static KeyFromEnvironment environmentKey(Map<String, String> env) {
        if (env.containsKey(ENV_KEY)) {
            String value = env.get(ENV_KEY);
            if (value == null || value.trim().isEmpty()) {
                throw new IllegalArgumentException(ENV_KEY + " is blank; it must hold an unsigned 64-bit decimal"
                        + " integer");
            }
            return new KeyFromEnvironment(parseEnvironmentKey(value, ENV_KEY, null), ENV_KEY);
        }
        if (env.containsKey(ENV_KEY_FILE)) {
            String file = env.get(ENV_KEY_FILE);
            if (file == null || file.trim().isEmpty()) {
                throw new IllegalArgumentException(ENV_KEY_FILE + " is blank; it must name a file holding the"
                        + " watermark key");
            }
            String path = file.trim();
            String text;
            try {
                text = Files.readString(Path.of(path), StandardCharsets.UTF_8).trim();
            } catch (IOException | RuntimeException e) {
                throw new IllegalArgumentException(ENV_KEY_FILE + " names a file that cannot be read: " + path, e);
            }
            if (text.isEmpty()) {
                throw new IllegalArgumentException(ENV_KEY_FILE + " names an empty file: " + path);
            }
            return new KeyFromEnvironment(parseEnvironmentKey(text, ENV_KEY_FILE, path), ENV_KEY_FILE);
        }
        return null;
    }

    /** {@code text} as an unsigned 64-bit decimal, refused naming {@code source} without repeating the text. */
    private static long parseEnvironmentKey(String text, String source, String file) {
        try {
            return Long.parseUnsignedLong(text.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(file == null
                    ? source + " must be an unsigned 64-bit decimal integer"
                    : source + " names a file that does not hold an unsigned 64-bit decimal integer: " + file, e);
        }
    }

    /** Logs once that the key came from the configuration property, which put it on the command line. */
    private static void warnCommandLineKeyOnce() {
        if (COMMAND_LINE_KEY_WARNED.compareAndSet(false, true)) {
            LOGGER.warn("the watermark key comes from the watermark configuration (the --watermark-config flag or"
                    + " the {} system property), which puts it on the process command line and in shell history;"
                    + " prefer the {} or {} environment variable", PROP_CONFIG, ENV_KEY, ENV_KEY_FILE);
        }
    }

    private static long parseKey(Object value) {
        if (value instanceof Long || value instanceof Integer || value instanceof BigInteger) {
            BigInteger big = value instanceof BigInteger ? (BigInteger) value : BigInteger.valueOf(((Number) value).longValue());
            if (big.signum() < 0 || big.compareTo(MAX_UNSIGNED_64) > 0) {
                throw new IllegalArgumentException("watermark config key must be an unsigned 64-bit integer");
            }
            return big.longValue();
        }
        if (value instanceof String) {
            try {
                return Long.parseUnsignedLong(((String) value).trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("watermark config key must be an unsigned 64-bit decimal"
                        + " integer", e);
            }
        }
        throw new IllegalArgumentException("watermark config key must be an unsigned 64-bit integer, got "
                + describeType(value));
    }

    /** The type of {@code value} without its text, for errors about a secret. */
    private static String describeType(Object value) {
        return value == null ? "null" : value.getClass().getSimpleName();
    }

    private static String string(Object value, String field) {
        if (!(value instanceof String)) {
            throw new IllegalArgumentException("watermark config " + field + " must be a string, got "
                    + describe(value));
        }
        return (String) value;
    }

    private static double number(Object value, String field) {
        if (!(value instanceof Number)) {
            throw new IllegalArgumentException("watermark config " + field + " must be a number, got "
                    + describe(value));
        }
        double v = ((Number) value).doubleValue();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            throw new IllegalArgumentException("watermark config " + field + " must be finite, got " + v);
        }
        return v;
    }

    private static int whole(Object value, String field) {
        double v = number(value, field);
        if (v != Math.rint(v) || v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("watermark config " + field + " must be a whole number, got "
                    + describe(value));
        }
        return (int) v;
    }

    private static String describe(Object value) {
        return value == null ? "null" : value.getClass().getSimpleName() + " " + value;
    }

    /** The secret key as its unsigned 64-bit value. */
    public long key() {
        return key;
    }

    /** The algorithm. */
    public WatermarkAlgorithm algorithm() {
        return algorithm;
    }

    /** The probability that a dual-key token is drawn under key B ({@code alpha}); key A takes the rest. */
    public double alpha() {
        return alpha;
    }

    /** The number of prior tokens the pseudorandom function keys on. */
    public int contextWidth() {
        return contextWidth;
    }

    /** Which history the generation repeated-context search scans. */
    public WatermarkContextScope deduplicateContexts() {
        return deduplicateContexts;
    }

    /** The most recent history positions the search scans, or null for the whole scope. */
    public Integer deduplicateContextsMaxHistory() {
        return deduplicateContextsMaxHistory;
    }

    /** The pseudorandom function. */
    public WatermarkPrfName prf() {
        return prf;
    }

    /** Whether a draft stage may leave its tokens unwatermarked ({@code allow_target_only_watermarking}). */
    public boolean allowTargetOnlyWatermarking() {
        return allowTargetOnlyWatermarking;
    }

    /** Whether the algorithm separates the draft and target roles ({@link WatermarkAlgorithm#DUAL_KEY_GUMBEL}). */
    public boolean supportsSpeculativeDecoding() {
        return algorithm == WatermarkAlgorithm.DUAL_KEY_GUMBEL;
    }

    /**
     * The watermarker of the target role under speculative decoding: the key-B watermarker of a dual-key
     * configuration, or the single key itself (vLLM's {@code create_speculative_target_watermarker}). The target
     * samples every token it selects under this role, so a dual-key configuration does not route here.
     */
    public Watermarker targetWatermarker() {
        long roleKey = algorithm == WatermarkAlgorithm.DUAL_KEY_GUMBEL ? keyB(key) : key;
        return new GumbelWatermarker(roleKey, contextWidth, prf);
    }

    /**
     * The watermarker of the draft role under speculative decoding (vLLM's
     * {@code create_speculative_draft_watermarker}): the key-A watermarker of a dual-key configuration, or null when
     * the algorithm has a single key and {@link #allowTargetOnlyWatermarking()} leaves the draft tokens
     * unwatermarked.
     *
     * @throws IllegalArgumentException when a single-key configuration does not allow target-only watermarking
     */
    public Watermarker draftWatermarker() {
        if (algorithm == WatermarkAlgorithm.DUAL_KEY_GUMBEL) {
            return new GumbelWatermarker(keyA(key), contextWidth, prf);
        }
        if (!allowTargetOnlyWatermarking) {
            throw new IllegalArgumentException("watermarking: the '" + algorithm.label() + "' algorithm does not"
                    + " support speculative decoding; set allow_target_only_watermarking=true to leave draft tokens"
                    + " unwatermarked");
        }
        return null;
    }

    /**
     * The key of a dual-key role: the first 8 bytes of {@code SHA-256(domain + key as 8 big-endian bytes)}, as an
     * unsigned 64-bit value. {@code domain} is {@code "key_a"} or {@code "key_b"}, matching vLLM's
     * {@code derive_watermark_key}.
     */
    public static long deriveKey(long key, String domain) {
        Objects.requireNonNull(domain, "domain");
        MessageDigest sha256;
        try {
            sha256 = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
        sha256.update(domain.getBytes(StandardCharsets.UTF_8));
        byte[] keyBytes = new byte[8];
        for (int i = 0; i < 8; i++) {
            keyBytes[i] = (byte) (key >>> (56 - 8 * i));
        }
        sha256.update(keyBytes);
        byte[] digest = sha256.digest();
        long derived = 0L;
        for (int i = 0; i < 8; i++) {
            derived = (derived << 8) | (digest[i] & 0xffL);
        }
        return derived;
    }

    /** The key of the dual-key A role. */
    public static long keyA(long key) {
        return deriveKey(key, KEY_A_DOMAIN);
    }

    /** The key of the dual-key B role. */
    public static long keyB(long key) {
        return deriveKey(key, KEY_B_DOMAIN);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof WatermarkConfig)) {
            return false;
        }
        WatermarkConfig other = (WatermarkConfig) o;
        return key == other.key && Double.compare(alpha, other.alpha) == 0
                && contextWidth == other.contextWidth && algorithm == other.algorithm
                && deduplicateContexts == other.deduplicateContexts
                && Objects.equals(deduplicateContextsMaxHistory, other.deduplicateContextsMaxHistory)
                && prf == other.prf && allowTargetOnlyWatermarking == other.allowTargetOnlyWatermarking;
    }

    @Override
    public int hashCode() {
        return Objects.hash(key, algorithm, alpha, contextWidth, deduplicateContexts,
                deduplicateContextsMaxHistory, prf, allowTargetOnlyWatermarking);
    }

    /** The settings without the secret key. */
    @Override
    public String toString() {
        return "WatermarkConfig{" + algorithm.label() + ", alpha=" + alpha + ", contextWidth=" + contextWidth
                + ", deduplicateContexts=" + deduplicateContexts.label()
                + ", deduplicateContextsMaxHistory=" + deduplicateContextsMaxHistory
                + ", prf=" + prf.label() + ", allowTargetOnlyWatermarking=" + allowTargetOnlyWatermarking + '}';
    }

    /** Builds a {@link WatermarkConfig}; the key is required and every value is range-checked at {@link #build()}. */
    public static final class Builder {

        private long key;
        private boolean keySet;
        private WatermarkAlgorithm algorithm = WatermarkAlgorithm.GUMBEL;
        private double alpha = DEFAULT_ALPHA;
        private int contextWidth = DEFAULT_CONTEXT_WIDTH;
        private WatermarkContextScope deduplicateContexts = WatermarkContextScope.DEFAULT;
        private Integer deduplicateContextsMaxHistory = DEFAULT_DEDUPLICATE_MAX_HISTORY;
        private WatermarkPrfName prf = WatermarkPrfName.PHILOX;
        private boolean allowTargetOnlyWatermarking;

        private Builder() {
        }

        /** The secret key as an unsigned 64-bit bit pattern (any {@code long} is a valid 64-bit key). */
        public Builder key(long v) {
            this.key = v;
            this.keySet = true;
            return this;
        }

        public Builder algorithm(WatermarkAlgorithm v) {
            this.algorithm = Objects.requireNonNull(v, "algorithm");
            return this;
        }

        public Builder alpha(double v) {
            this.alpha = v;
            return this;
        }

        public Builder contextWidth(int v) {
            this.contextWidth = v;
            return this;
        }

        public Builder deduplicateContexts(WatermarkContextScope v) {
            this.deduplicateContexts = Objects.requireNonNull(v, "deduplicateContexts");
            return this;
        }

        /** The most recent history positions the search scans; null scans the whole scope. */
        public Builder deduplicateContextsMaxHistory(Integer v) {
            this.deduplicateContextsMaxHistory = v;
            return this;
        }

        public Builder prf(WatermarkPrfName v) {
            this.prf = Objects.requireNonNull(v, "prf");
            return this;
        }

        public Builder allowTargetOnlyWatermarking(boolean v) {
            this.allowTargetOnlyWatermarking = v;
            return this;
        }

        /**
         * @throws IllegalArgumentException when the key was not set, {@code alpha} is outside {@code [0, 1]}, the
         *                                  context width is below 1, or the history bound is below 1
         */
        public WatermarkConfig build() {
            if (!keySet) {
                throw new IllegalArgumentException("watermark config has no 'key'");
            }
            if (!(alpha >= 0.0 && alpha <= 1.0)) {
                throw new IllegalArgumentException("watermark alpha must be in [0, 1], got " + alpha);
            }
            if (contextWidth < 1) {
                throw new IllegalArgumentException("watermark context_width must be 1 or more, got "
                        + contextWidth);
            }
            if (deduplicateContextsMaxHistory != null && deduplicateContextsMaxHistory < 1) {
                throw new IllegalArgumentException("watermark deduplicate_contexts_max_history must be 1 or more"
                        + " (null scans the whole scope), got " + deduplicateContextsMaxHistory);
            }
            return new WatermarkConfig(this);
        }
    }
}
