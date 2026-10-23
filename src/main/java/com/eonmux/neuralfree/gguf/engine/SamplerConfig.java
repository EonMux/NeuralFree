package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;
import java.util.Properties;

/**
 * Token sampling settings, applied in llama.cpp's default chain order (llama.cpp common/common.h
 * {@code common_params_sampling::samplers}): penalties, top-k, typical-p, top-p, min-p, temperature, then a
 * seeded categorical draw. Defaults are llama.cpp's. Immutable.
 *
 * <p>Disabled values: {@code topK <= 0}, {@code topP == 1}, {@code minP == 0}, {@code typicalP == 1},
 * {@code repeatPenalty == 1}, {@code presencePenalty == 0}, {@code frequencyPenalty == 0},
 * {@code penaltyLastN == 0} (all penalties off). {@code penaltyLastN == -1} penalizes over the whole context.
 * Sampling is greedy when {@link #greedy()} is set or {@code temperature <= 0}.
 */
public final class SamplerConfig {

    /** Seed value meaning "draw a fresh random seed per generation". */
    public static final long RANDOM_SEED = -1L;
    /** {@code penaltyLastN} value meaning "the whole context". */
    public static final int PENALTY_WHOLE_CONTEXT = -1;

    public static final String PROP_TEMPERATURE = "neuralfree.temp";
    public static final String PROP_TOP_K = "neuralfree.top.k";
    public static final String PROP_TOP_P = "neuralfree.top.p";
    public static final String PROP_MIN_P = "neuralfree.min.p";
    public static final String PROP_TYPICAL_P = "neuralfree.typical.p";
    public static final String PROP_REPEAT_PENALTY = "neuralfree.repeat.penalty";
    public static final String PROP_PRESENCE_PENALTY = "neuralfree.presence.penalty";
    public static final String PROP_FREQUENCY_PENALTY = "neuralfree.frequency.penalty";
    public static final String PROP_PENALTY_LAST_N = "neuralfree.repeat.last.n";
    public static final String PROP_SEED = "neuralfree.seed";
    public static final String PROP_GREEDY = "neuralfree.greedy";

    private static final SamplerConfig DEFAULTS = builder().build();

    private final float temperature;
    private final int topK;
    private final float topP;
    private final float minP;
    private final float typicalP;
    private final float repeatPenalty;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final int penaltyLastN;
    private final long seed;
    private final boolean greedy;

    private SamplerConfig(Builder b) {
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.minP = b.minP;
        this.typicalP = b.typicalP;
        this.repeatPenalty = b.repeatPenalty;
        this.presencePenalty = b.presencePenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.penaltyLastN = b.penaltyLastN;
        this.seed = b.seed;
        this.greedy = b.greedy;
    }

    /** llama.cpp defaults: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, penalties off, last-n 64, random seed. */
    public static SamplerConfig defaults() {
        return DEFAULTS;
    }

    /** Greedy (argmax) sampling with every other stage disabled. */
    public static SamplerConfig greedyDecoding() {
        return builder().greedy(true).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Settings from a property set; absent keys keep the llama.cpp defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static SamplerConfig fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder d = builder();
        return builder()
                .temperature(EngineConfig.floatProperty(p, PROP_TEMPERATURE, d.temperature))
                .topK(EngineConfig.intProperty(p, PROP_TOP_K, d.topK))
                .topP(EngineConfig.floatProperty(p, PROP_TOP_P, d.topP))
                .minP(EngineConfig.floatProperty(p, PROP_MIN_P, d.minP))
                .typicalP(EngineConfig.floatProperty(p, PROP_TYPICAL_P, d.typicalP))
                .repeatPenalty(EngineConfig.floatProperty(p, PROP_REPEAT_PENALTY, d.repeatPenalty))
                .presencePenalty(EngineConfig.floatProperty(p, PROP_PRESENCE_PENALTY, d.presencePenalty))
                .frequencyPenalty(EngineConfig.floatProperty(p, PROP_FREQUENCY_PENALTY, d.frequencyPenalty))
                .penaltyLastN(EngineConfig.intProperty(p, PROP_PENALTY_LAST_N, d.penaltyLastN))
                .seed(EngineConfig.longProperty(p, PROP_SEED, d.seed))
                .greedy(EngineConfig.booleanProperty(p, PROP_GREEDY, d.greedy))
                .build();
    }

    public float temperature() { return temperature; }
    public int topK() { return topK; }
    public float topP() { return topP; }
    public float minP() { return minP; }
    public float typicalP() { return typicalP; }
    public float repeatPenalty() { return repeatPenalty; }
    public float presencePenalty() { return presencePenalty; }
    public float frequencyPenalty() { return frequencyPenalty; }
    /** Tokens the penalties look back over: 0 disables them, {@link #PENALTY_WHOLE_CONTEXT} means all. */
    public int penaltyLastN() { return penaltyLastN; }
    /** The RNG seed, or {@link #RANDOM_SEED}. */
    public long seed() { return seed; }
    /** Whether greedy decoding was requested explicitly. */
    public boolean greedy() { return greedy; }

    /** True when the chain reduces to argmax: {@link #greedy()} or a temperature {@code <= 0}. */
    public boolean isGreedy() {
        return greedy || temperature <= 0f;
    }

    /** True when any repetition, presence or frequency penalty applies (llama.cpp penalties sampler). */
    public boolean penaltiesEnabled() {
        return penaltyLastN != 0
                && (repeatPenalty != 1f || presencePenalty != 0f || frequencyPenalty != 0f);
    }

    /** True when the seed is fixed. */
    public boolean seeded() {
        return seed != RANDOM_SEED;
    }

    public Builder toBuilder() {
        return builder().temperature(temperature).topK(topK).topP(topP).minP(minP).typicalP(typicalP)
                .repeatPenalty(repeatPenalty).presencePenalty(presencePenalty).frequencyPenalty(frequencyPenalty)
                .penaltyLastN(penaltyLastN).seed(seed).greedy(greedy);
    }

    /** Copy with another seed. */
    public SamplerConfig withSeed(long newSeed) {
        return toBuilder().seed(newSeed).build();
    }

    @Override
    public String toString() {
        return "SamplerConfig{" + (isGreedy() ? "greedy, " : "temp=" + temperature + ", ") + "topK=" + topK
                + ", topP=" + topP + ", minP=" + minP + ", typicalP=" + typicalP + ", repeat=" + repeatPenalty
                + ", presence=" + presencePenalty + ", frequency=" + frequencyPenalty + ", lastN=" + penaltyLastN
                + ", seed=" + (seeded() ? String.valueOf(seed) : "random") + '}';
    }

    public static final class Builder {
        private float temperature = 0.8f;
        private int topK = 40;
        private float topP = 0.95f;
        private float minP = 0.05f;
        private float typicalP = 1.0f;
        private float repeatPenalty = 1.0f;
        private float presencePenalty = 0.0f;
        private float frequencyPenalty = 0.0f;
        private int penaltyLastN = 64;
        private long seed = RANDOM_SEED;
        private boolean greedy;

        private Builder() {
        }

        public Builder temperature(float v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(float v) { this.topP = v; return this; }
        public Builder minP(float v) { this.minP = v; return this; }
        public Builder typicalP(float v) { this.typicalP = v; return this; }
        public Builder repeatPenalty(float v) { this.repeatPenalty = v; return this; }
        public Builder presencePenalty(float v) { this.presencePenalty = v; return this; }
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return this; }
        public Builder penaltyLastN(int v) { this.penaltyLastN = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder greedy(boolean v) { this.greedy = v; return this; }

        /** @throws IllegalArgumentException naming the setting when a value is out of range */
        public SamplerConfig build() {
            requireFinite("temperature", temperature);
            requireUnit("topP", topP);
            requireUnit("minP", minP);
            requireUnit("typicalP", typicalP);
            requireFinite("repeatPenalty", repeatPenalty);
            if (repeatPenalty <= 0f) {
                throw new IllegalArgumentException("repeatPenalty must be > 0 (1 = disabled), got " + repeatPenalty);
            }
            requireFinite("presencePenalty", presencePenalty);
            requireFinite("frequencyPenalty", frequencyPenalty);
            if (penaltyLastN < PENALTY_WHOLE_CONTEXT) {
                throw new IllegalArgumentException(
                        "penaltyLastN must be >= -1 (-1 = whole context, 0 = disabled), got " + penaltyLastN);
            }
            if (seed < RANDOM_SEED) {
                throw new IllegalArgumentException("seed must be >= 0, or -1 for a random seed, got " + seed);
            }
            return new SamplerConfig(this);
        }

        private static void requireFinite(String name, float v) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException(name + " must be finite, got " + v);
            }
        }

        private static void requireUnit(String name, float v) {
            if (!(v >= 0f && v <= 1f)) {
                throw new IllegalArgumentException(name + " must be in [0, 1], got " + v);
            }
        }
    }
}
