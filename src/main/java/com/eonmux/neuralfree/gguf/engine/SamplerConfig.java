package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Token sampling settings, applied in a llama.cpp-configurable chain order. The order defaults to llama.cpp's
 * {@code common_params_sampling::samplers} (llama.cpp common/common.h:265-273,
 * {@link StageKind#DEFAULT_ORDER}), which with the stages SA-2 adds off is the fixed chain the fixed dist stage
 * follows: penalties, top-k, typical-p, top-p, min-p, temperature, then a seeded categorical draw (the bias stage
 * runs before the chain). Defaults are llama.cpp's. Immutable.
 *
 * <p>Disabled values: {@code topK <= 0}, {@code topP == 1}, {@code minP == 0}, {@code typicalP == 1},
 * {@code repeatPenalty == 1}, {@code presencePenalty == 0}, {@code frequencyPenalty == 0},
 * {@code penaltyLastN == 0} (all penalties off). {@code penaltyLastN == -1} penalizes over the whole context.
 * Sampling is greedy when {@link #greedy()} is set or {@code temperature <= 0}.
 *
 * <p>Each {@link Setting} is either explicit (given by a property or a {@link Builder} setter) or a fallback (a
 * default, or a {@link Builder} {@code fallback*} setter). A model's sampling defaults replace only fallbacks
 * ({@link #withModelDefaults}), as llama.cpp applies {@code general.sampling.*} only to parameters the user did not set
 * ({@code common_params_sampling::user_sampling_config}, common/common.cpp:1088-1147).
 */
public final class SamplerConfig {

    /** The settings a model's sampling defaults can supply. */
    public enum Setting {
        TEMPERATURE, TOP_K, TOP_P, MIN_P, TYPICAL_P, REPEAT_PENALTY, PENALTY_LAST_N, PRESENCE_PENALTY,
        FREQUENCY_PENALTY, SEQUENCE
    }

    /** Seed value meaning "draw a fresh random seed per generation". */
    public static final long RANDOM_SEED = -1L;
    /** {@code penaltyLastN} value meaning "the whole context". */
    public static final int PENALTY_WHOLE_CONTEXT = -1;

    public static final String PROP_TEMPERATURE = "neuralfree.temp";
    public static final String PROP_TOP_K = "neuralfree.top.k";
    public static final String PROP_TOP_P = "neuralfree.top.p";
    public static final String PROP_MIN_P = "neuralfree.min.p";
    public static final String PROP_MIN_KEEP = "neuralfree.min.keep";
    public static final String PROP_TYPICAL_P = "neuralfree.typical.p";
    public static final String PROP_REPEAT_PENALTY = "neuralfree.repeat.penalty";
    public static final String PROP_PRESENCE_PENALTY = "neuralfree.presence.penalty";
    public static final String PROP_FREQUENCY_PENALTY = "neuralfree.frequency.penalty";
    public static final String PROP_PENALTY_LAST_N = "neuralfree.repeat.last.n";
    public static final String PROP_SEED = "neuralfree.seed";
    public static final String PROP_GREEDY = "neuralfree.greedy";
    public static final String PROP_SAMPLERS = "neuralfree.samplers";

    private static final SamplerConfig DEFAULTS = builder().build();

    private final float temperature;
    private final int topK;
    private final float topP;
    private final float minP;
    private final int minKeep;
    private final float typicalP;
    private final float repeatPenalty;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final int penaltyLastN;
    private final long seed;
    private final boolean greedy;
    private final List<StageKind> order;
    private final Set<Setting> explicit;

    private SamplerConfig(Builder b) {
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.minP = b.minP;
        this.minKeep = b.minKeep;
        this.typicalP = b.typicalP;
        this.repeatPenalty = b.repeatPenalty;
        this.presencePenalty = b.presencePenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.penaltyLastN = b.penaltyLastN;
        this.seed = b.seed;
        this.greedy = b.greedy;
        this.order = b.order;
        this.explicit = Collections.unmodifiableSet(EnumSet.copyOf(b.explicit));
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
     * Settings from a property set; absent keys keep the llama.cpp defaults, and every present key makes its
     * setting explicit.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static SamplerConfig fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder b = builder();
        floatProperty(p, PROP_TEMPERATURE, b::temperature);
        intProperty(p, PROP_TOP_K, b::topK);
        floatProperty(p, PROP_TOP_P, b::topP);
        floatProperty(p, PROP_MIN_P, b::minP);
        intProperty(p, PROP_MIN_KEEP, b::minKeep);
        floatProperty(p, PROP_TYPICAL_P, b::typicalP);
        floatProperty(p, PROP_REPEAT_PENALTY, b::repeatPenalty);
        floatProperty(p, PROP_PRESENCE_PENALTY, b::presencePenalty);
        floatProperty(p, PROP_FREQUENCY_PENALTY, b::frequencyPenalty);
        intProperty(p, PROP_PENALTY_LAST_N, b::penaltyLastN);
        String samplers = EngineConfig.stringProperty(p, PROP_SAMPLERS);
        if (samplers != null) {
            try {
                b.samplers(StageKind.parseOrder(samplers));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(PROP_SAMPLERS + ": " + e.getMessage(), e);
            }
        }
        return b.seed(EngineConfig.longProperty(p, PROP_SEED, b.seed))
                .greedy(EngineConfig.booleanProperty(p, PROP_GREEDY, b.greedy))
                .build();
    }

    /** Receives a float property value. */
    private interface FloatSetter {
        void set(float v);
    }

    /** Receives an int property value. */
    private interface IntSetter {
        void set(int v);
    }

    /** Passes the value of {@code key} to {@code setter} when the property is set (non-blank). */
    private static void floatProperty(Properties p, String key, FloatSetter setter) {
        if (EngineConfig.stringProperty(p, key) != null) {
            setter.set(EngineConfig.floatProperty(p, key, 0f));
        }
    }

    /** Passes the value of {@code key} to {@code setter} when the property is set (non-blank). */
    private static void intProperty(Properties p, String key, IntSetter setter) {
        if (EngineConfig.stringProperty(p, key) != null) {
            setter.set(EngineConfig.intProperty(p, key, 0));
        }
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
    /** Fewest candidates to keep after top-k, top-p, min-p and typical-p (llama.cpp's {@code min_keep}). */
    public int minKeep() { return minKeep; }
    /** The RNG seed, or {@link #RANDOM_SEED}. */
    public long seed() { return seed; }
    /** Whether greedy decoding was requested explicitly. */
    public boolean greedy() { return greedy; }
    /** llama.cpp's chain order (unmodifiable); {@link StageKind#DEFAULT_ORDER} unless configured otherwise. */
    public List<StageKind> order() { return order; }

    /** The settings given explicitly (unmodifiable); a model's sampling defaults never replace them. */
    public Set<Setting> explicit() { return explicit; }

    /** Whether {@code s} was given explicitly. */
    public boolean isExplicit(Setting s) {
        return explicit.contains(Objects.requireNonNull(s, "setting"));
    }

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

    /** A builder holding these values, with the same settings explicit. */
    public Builder toBuilder() {
        Builder b = builder();
        b.temperature = temperature;
        b.topK = topK;
        b.topP = topP;
        b.minP = minP;
        b.typicalP = typicalP;
        b.repeatPenalty = repeatPenalty;
        b.presencePenalty = presencePenalty;
        b.frequencyPenalty = frequencyPenalty;
        b.penaltyLastN = penaltyLastN;
        b.seed = seed;
        b.greedy = greedy;
        b.order = order;
        b.explicit.addAll(explicit);
        return b;
    }

    /** Copy with another seed. */
    public SamplerConfig withSeed(long newSeed) {
        return toBuilder().seed(newSeed).build();
    }

    /**
     * Copy in which every setting that is not explicit takes the model's default when {@code defaults} has one; the
     * taken values stay fallbacks. Returns this config when {@code defaults} is empty.
     *
     * @throws IllegalArgumentException when a taken value is out of range ({@link Builder#build()})
     */
    public SamplerConfig withModelDefaults(SamplingDefaults defaults) {
        Objects.requireNonNull(defaults, "defaults");
        if (defaults.isEmpty()) {
            return this;
        }
        Builder b = toBuilder();
        if (defaults.temp() != null && !explicit.contains(Setting.TEMPERATURE)) {
            b.fallbackTemperature(defaults.temp());
        }
        if (defaults.topK() != null && !explicit.contains(Setting.TOP_K)) {
            b.fallbackTopK(defaults.topK());
        }
        if (defaults.topP() != null && !explicit.contains(Setting.TOP_P)) {
            b.fallbackTopP(defaults.topP());
        }
        if (defaults.minP() != null && !explicit.contains(Setting.MIN_P)) {
            b.fallbackMinP(defaults.minP());
        }
        if (defaults.typicalP() != null && !explicit.contains(Setting.TYPICAL_P)) {
            b.fallbackTypicalP(defaults.typicalP());
        }
        if (defaults.penaltyRepeat() != null && !explicit.contains(Setting.REPEAT_PENALTY)) {
            b.fallbackRepeatPenalty(defaults.penaltyRepeat());
        }
        if (defaults.penaltyLastN() != null && !explicit.contains(Setting.PENALTY_LAST_N)) {
            b.fallbackPenaltyLastN(defaults.penaltyLastN());
        }
        if (defaults.presencePenalty() != null && !explicit.contains(Setting.PRESENCE_PENALTY)) {
            b.fallbackPresencePenalty(defaults.presencePenalty());
        }
        if (defaults.frequencyPenalty() != null && !explicit.contains(Setting.FREQUENCY_PENALTY)) {
            b.fallbackFrequencyPenalty(defaults.frequencyPenalty());
        }
        if (defaults.sequence() != null && !explicit.contains(Setting.SEQUENCE)) {
            b.fallbackSamplers(defaults.sequence());
        }
        return b.build();
    }

    @Override
    public String toString() {
        return "SamplerConfig{" + (isGreedy() ? "greedy, " : "temp=" + temperature + ", ") + "topK=" + topK
                + ", topP=" + topP + ", minP=" + minP + ", typicalP=" + typicalP + ", repeat=" + repeatPenalty
                + ", presence=" + presencePenalty + ", frequency=" + frequencyPenalty + ", lastN=" + penaltyLastN
                + ", seed=" + (seeded() ? String.valueOf(seed) : "random") + '}';
    }

    /**
     * Each setter makes its setting explicit. Each {@code fallback*} setter sets the value and leaves the setting a
     * fallback, which a model's sampling defaults replace ({@link SamplerConfig#withModelDefaults}).
     */
    public static final class Builder {
        private float temperature = 0.8f;
        private int topK = 40;
        private float topP = 0.95f;
        private float minP = 0.05f;
        private int minKeep = 0;
        private float typicalP = 1.0f;
        private float repeatPenalty = 1.0f;
        private float presencePenalty = 0.0f;
        private float frequencyPenalty = 0.0f;
        private int penaltyLastN = 64;
        private long seed = RANDOM_SEED;
        private boolean greedy;
        private List<StageKind> order = StageKind.DEFAULT_ORDER;
        private final EnumSet<Setting> explicit = EnumSet.noneOf(Setting.class);

        private Builder() {
        }

        public Builder temperature(float v) { this.temperature = v; return mark(Setting.TEMPERATURE); }
        public Builder topK(int v) { this.topK = v; return mark(Setting.TOP_K); }
        public Builder topP(float v) { this.topP = v; return mark(Setting.TOP_P); }
        public Builder minP(float v) { this.minP = v; return mark(Setting.MIN_P); }
        public Builder typicalP(float v) { this.typicalP = v; return mark(Setting.TYPICAL_P); }
        public Builder repeatPenalty(float v) { this.repeatPenalty = v; return mark(Setting.REPEAT_PENALTY); }
        public Builder presencePenalty(float v) { this.presencePenalty = v; return mark(Setting.PRESENCE_PENALTY); }
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return mark(Setting.FREQUENCY_PENALTY); }
        public Builder penaltyLastN(int v) { this.penaltyLastN = v; return mark(Setting.PENALTY_LAST_N); }
        public Builder minKeep(int v) { this.minKeep = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder greedy(boolean v) { this.greedy = v; return this; }

        /** llama.cpp's chain order ({@code neuralfree.samplers}); the kind, not the fixed bias or dist stage. */
        public Builder samplers(List<StageKind> order) {
            this.order = StageKind.parseOrder(joined(order));
            return mark(Setting.SEQUENCE);
        }

        /** {@link #samplers(List)} for a '{@code ;}'-separated order. */
        public Builder samplers(String joined) {
            this.order = StageKind.parseOrder(joined);
            return mark(Setting.SEQUENCE);
        }

        /** {@link #fallbackSamplers(List)} taking the order and leaving {@link Setting#SEQUENCE} a fallback. */
        public Builder fallbackSamplers(List<StageKind> order) {
            this.order = StageKind.parseOrder(joined(order));
            return unmark(Setting.SEQUENCE);
        }

        /** {@link #fallbackSamplers(List)} for a '{@code ;}'-separated order. */
        public Builder fallbackSamplers(String joined) {
            this.order = StageKind.parseOrder(joined);
            return unmark(Setting.SEQUENCE);
        }

        public Builder fallbackTemperature(float v) { this.temperature = v; return unmark(Setting.TEMPERATURE); }
        public Builder fallbackTopK(int v) { this.topK = v; return unmark(Setting.TOP_K); }
        public Builder fallbackTopP(float v) { this.topP = v; return unmark(Setting.TOP_P); }
        public Builder fallbackMinP(float v) { this.minP = v; return unmark(Setting.MIN_P); }
        public Builder fallbackTypicalP(float v) { this.typicalP = v; return unmark(Setting.TYPICAL_P); }
        public Builder fallbackRepeatPenalty(float v) { this.repeatPenalty = v; return unmark(Setting.REPEAT_PENALTY); }
        public Builder fallbackPresencePenalty(float v) {
            this.presencePenalty = v;
            return unmark(Setting.PRESENCE_PENALTY);
        }
        public Builder fallbackFrequencyPenalty(float v) {
            this.frequencyPenalty = v;
            return unmark(Setting.FREQUENCY_PENALTY);
        }
        public Builder fallbackPenaltyLastN(int v) { this.penaltyLastN = v; return unmark(Setting.PENALTY_LAST_N); }

        private static String joined(List<StageKind> order) {
            Objects.requireNonNull(order, "sampler order");
            if (order.isEmpty()) {
                throw new IllegalArgumentException("sampler order is empty");
            }
            StringBuilder s = new StringBuilder();
            for (StageKind kind : order) {
                if (s.length() > 0) {
                    s.append(';');
                }
                s.append(kind.label());
            }
            return s.toString();
        }

        private Builder mark(Setting s) {
            explicit.add(s);
            return this;
        }

        private Builder unmark(Setting s) {
            explicit.remove(s);
            return this;
        }

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
            if (minKeep < 0) {
                throw new IllegalArgumentException(PROP_MIN_KEEP + " must be >= 0, got " + minKeep);
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
