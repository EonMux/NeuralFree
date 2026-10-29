package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;

import java.util.ArrayList;
import java.util.Arrays;
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
        FREQUENCY_PENALTY, SEQUENCE, MIROSTAT, MIROSTAT_TAU, MIROSTAT_ETA, XTC_PROBABILITY, XTC_THRESHOLD
    }

    /** Seed value meaning "draw a fresh random seed per generation". */
    public static final long RANDOM_SEED = -1L;
    /** {@code penaltyLastN} value meaning "the whole context". */
    public static final int PENALTY_WHOLE_CONTEXT = -1;
    /** llama.cpp's default DRY sequence breakers ({@code "\n", ":", '"', "*"}). */
    public static final List<String> DRY_DEFAULT_BREAKERS =
            Collections.unmodifiableList(Arrays.asList("\n", ":", "\"", "*"));
    /** llama.cpp's default DRY base ({@code dry_base}). */
    public static final float DRY_DEFAULT_BASE = 1.75f;
    /** llama.cpp's default DRY allowed length ({@code dry_allowed_length}). */
    public static final int DRY_DEFAULT_ALLOWED_LENGTH = 2;
    /** llama.cpp's default DRY window ({@code dry_penalty_last_n}). */
    public static final int DRY_DEFAULT_LAST_N = 64;
    /** llama.cpp's default XTC threshold ({@code xtc_threshold}; above 0.5 is the disabled marker). */
    public static final float XTC_DEFAULT_THRESHOLD = 0.10f;
    /** llama.cpp's default mirostat target entropy ({@code mirostat_tau}). */
    public static final float MIROSTAT_DEFAULT_TAU = 5.0f;
    /** llama.cpp's default mirostat learning rate ({@code mirostat_eta}). */
    public static final float MIROSTAT_DEFAULT_ETA = 0.10f;
    /** Mirostat disabled ({@code mirostat = 0}). */
    public static final int MIROSTAT_OFF = 0;

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
    // llama.cpp's sampler flags, as SA-2's stage parameters (the only one is a repeated CLI flag, folded here into a
    // ','-separated list: a sequence breaker keeps its ',' characters, the list boundaries cannot)
    public static final String PROP_DRY_MULTIPLIER = "neuralfree.dry.multiplier";
    public static final String PROP_DRY_BASE = "neuralfree.dry.base";
    public static final String PROP_DRY_ALLOWED_LENGTH = "neuralfree.dry.allowed.length";
    public static final String PROP_DRY_PENALTY_LAST_N = "neuralfree.dry.penalty.last.n";
    public static final String PROP_DRY_SEQUENCE_BREAKERS = "neuralfree.dry.sequence.breakers";
    public static final String PROP_TOP_N_SIGMA = "neuralfree.top.n.sigma";
    public static final String PROP_XTC_PROBABILITY = "neuralfree.xtc.probability";
    public static final String PROP_XTC_THRESHOLD = "neuralfree.xtc.threshold";
    public static final String PROP_MIROSTAT = "neuralfree.mirostat";
    public static final String PROP_MIROSTAT_TAU = "neuralfree.mirostat.tau";
    public static final String PROP_MIROSTAT_ETA = "neuralfree.mirostat.eta";
    public static final String PROP_DYNATEMP_RANGE = "neuralfree.dynatemp.range";
    public static final String PROP_DYNATEMP_EXPONENT = "neuralfree.dynatemp.exponent";
    public static final String PROP_ADAPTIVE_TARGET = "neuralfree.adaptive.target";
    public static final String PROP_ADAPTIVE_DECAY = "neuralfree.adaptive.decay";

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
    private final float dryMultiplier;
    private final float dryBase;
    private final int dryAllowedLength;
    private final int dryPenaltyLastN;
    private final List<String> drySequenceBreakers;
    private final float topNSigma;
    private final float xtcProbability;
    private final float xtcThreshold;
    private final int mirostat;
    private final float mirostatTau;
    private final float mirostatEta;
    private final float dynatempRange;
    private final float dynatempExponent;
    private final float adaptiveTarget;
    private final float adaptiveDecay;
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
        this.dryMultiplier = b.dryMultiplier;
        this.dryBase = b.dryBase;
        this.dryAllowedLength = b.dryAllowedLength;
        this.dryPenaltyLastN = b.dryPenaltyLastN;
        this.drySequenceBreakers = b.drySequenceBreakers;
        this.topNSigma = b.topNSigma;
        this.xtcProbability = b.xtcProbability;
        this.xtcThreshold = b.xtcThreshold;
        this.mirostat = b.mirostat;
        this.mirostatTau = b.mirostatTau;
        this.mirostatEta = b.mirostatEta;
        this.dynatempRange = b.dynatempRange;
        this.dynatempExponent = b.dynatempExponent;
        this.adaptiveTarget = b.adaptiveTarget;
        this.adaptiveDecay = b.adaptiveDecay;
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
        floatProperty(p, PROP_DRY_MULTIPLIER, b::dryMultiplier);
        floatProperty(p, PROP_DRY_BASE, b::dryBase);
        intProperty(p, PROP_DRY_ALLOWED_LENGTH, b::dryAllowedLength);
        intProperty(p, PROP_DRY_PENALTY_LAST_N, b::dryPenaltyLastN);
        // the breakers keep their whitespace: a leading newline is the most common breaker
        String breakers = p.getProperty(PROP_DRY_SEQUENCE_BREAKERS);
        if (breakers != null && !breakers.isBlank()) {
            List<String> list = new ArrayList<>();
            for (String part : breakers.split(",", -1)) {
                if (part.isEmpty()) {
                    throw new IllegalArgumentException(
                            PROP_DRY_SEQUENCE_BREAKERS + " has an empty breaker in '" + breakers + "'");
                }
                list.add(part);
            }
            b.drySequenceBreakers(list);
        }
        floatProperty(p, PROP_TOP_N_SIGMA, b::topNSigma);
        floatProperty(p, PROP_XTC_PROBABILITY, b::xtcProbability);
        floatProperty(p, PROP_XTC_THRESHOLD, b::xtcThreshold);
        intProperty(p, PROP_MIROSTAT, b::mirostat);
        floatProperty(p, PROP_MIROSTAT_TAU, b::mirostatTau);
        floatProperty(p, PROP_MIROSTAT_ETA, b::mirostatEta);
        floatProperty(p, PROP_DYNATEMP_RANGE, b::dynatempRange);
        floatProperty(p, PROP_DYNATEMP_EXPONENT, b::dynatempExponent);
        floatProperty(p, PROP_ADAPTIVE_TARGET, b::adaptiveTarget);
        floatProperty(p, PROP_ADAPTIVE_DECAY, b::adaptiveDecay);
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

    /** DRY repetition strength ({@code --dry-multiplier}); 0 disables the stage. */
    public float dryMultiplier() { return dryMultiplier; }
    /** DRY base of the exponential penalty ({@code --dry-base}); below 1 disables the stage. */
    public float dryBase() { return dryBase; }
    /** DRY repeated suffix length the penalty starts at ({@code --dry-allowed-length}). */
    public int dryAllowedLength() { return dryAllowedLength; }
    /** DRY window of accepted tokens ({@code --dry-penalty-last-n}); 0 disables the stage. */
    public int dryPenaltyLastN() { return dryPenaltyLastN; }
    /** DRY restart sequences ({@code --dry-sequence-breaker}, unmodifiable), llama.cpp's defaults unless set. */
    public List<String> drySequenceBreakers() { return drySequenceBreakers; }
    /** top-n-sigma's n ({@code --top-n-sigma}); 0 and below disables the stage. */
    public float topNSigma() { return topNSigma; }
    /** XTC's trim trigger probability ({@code --xtc-probability}); 0 disables the stage. */
    public float xtcProbability() { return xtcProbability; }
    /** XTC's probability threshold ({@code --xtc-threshold}); above 0.5 disables the stage. */
    public float xtcThreshold() { return xtcThreshold; }
    /** Mirostat version: {@link #MIROSTAT_OFF} (0), 1 or 2 ({@code --mirostat}). */
    public int mirostat() { return mirostat; }
    /** Mirostat target entropy ({@code --mirostat-ent}). */
    public float mirostatTau() { return mirostatTau; }
    /** Mirostat learning rate ({@code --mirostat-lr}). */
    public float mirostatEta() { return mirostatEta; }
    /** Dynamic temperature half-width ({@code --dynatemp-range}); 0 keeps the plain scaling. */
    public float dynatempRange() { return dynatempRange; }
    /** Dynamic temperature entropy exponent ({@code --dynatemp-exponent}). */
    public float dynatempExponent() { return dynatempExponent; }
    /** adaptive-p's target probability ({@code --adaptive-target}); negative disables the adaptation. */
    public float adaptiveTarget() { return adaptiveTarget; }
    /** adaptive-p's EMA decay ({@code --adaptive-decay}). */
    public float adaptiveDecay() { return adaptiveDecay; }

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
        b.dryMultiplier = dryMultiplier;
        b.dryBase = dryBase;
        b.dryAllowedLength = dryAllowedLength;
        b.dryPenaltyLastN = dryPenaltyLastN;
        b.drySequenceBreakers = drySequenceBreakers;
        b.topNSigma = topNSigma;
        b.xtcProbability = xtcProbability;
        b.xtcThreshold = xtcThreshold;
        b.mirostat = mirostat;
        b.mirostatTau = mirostatTau;
        b.mirostatEta = mirostatEta;
        b.dynatempRange = dynatempRange;
        b.dynatempExponent = dynatempExponent;
        b.adaptiveTarget = adaptiveTarget;
        b.adaptiveDecay = adaptiveDecay;
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
        if (defaults.mirostat() != null && !explicit.contains(Setting.MIROSTAT)) {
            b.fallbackMirostat(defaults.mirostat());
        }
        if (defaults.mirostatTau() != null && !explicit.contains(Setting.MIROSTAT_TAU)) {
            b.fallbackMirostatTau(defaults.mirostatTau());
        }
        if (defaults.mirostatEta() != null && !explicit.contains(Setting.MIROSTAT_ETA)) {
            b.fallbackMirostatEta(defaults.mirostatEta());
        }
        if (defaults.xtcProbability() != null && !explicit.contains(Setting.XTC_PROBABILITY)) {
            b.fallbackXtcProbability(defaults.xtcProbability());
        }
        if (defaults.xtcThreshold() != null && !explicit.contains(Setting.XTC_THRESHOLD)) {
            b.fallbackXtcThreshold(defaults.xtcThreshold());
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
        // llama.cpp's CLI defaults (common_params_sampling)
        private float dryMultiplier = 0.0f;
        private float dryBase = DRY_DEFAULT_BASE;
        private int dryAllowedLength = DRY_DEFAULT_ALLOWED_LENGTH;
        private int dryPenaltyLastN = DRY_DEFAULT_LAST_N;
        private List<String> drySequenceBreakers = DRY_DEFAULT_BREAKERS;
        private float topNSigma = 0.0f;
        private float xtcProbability = 0.0f;
        private float xtcThreshold = XTC_DEFAULT_THRESHOLD;
        private int mirostat = MIROSTAT_OFF;
        private float mirostatTau = MIROSTAT_DEFAULT_TAU;
        private float mirostatEta = MIROSTAT_DEFAULT_ETA;
        private float dynatempRange = 0.0f;
        private float dynatempExponent = 1.0f;
        private float adaptiveTarget = -1.0f;
        private float adaptiveDecay = 0.90f;
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

        /** DRY repetition strength ({@code neuralfree.dry.multiplier}); 0 disables the stage. */
        public Builder dryMultiplier(float v) { this.dryMultiplier = v; return this; }
        /** DRY base of the exponential penalty ({@code neuralfree.dry.base}); below 1 disables the stage. */
        public Builder dryBase(float v) { this.dryBase = v; return this; }
        /** DRY length the penalty starts at ({@code neuralfree.dry.allowed.length}). */
        public Builder dryAllowedLength(int v) { this.dryAllowedLength = v; return this; }
        /** DRY window of accepted tokens ({@code neuralfree.dry.penalty.last.n}); 0 disables the stage. */
        public Builder dryPenaltyLastN(int v) { this.dryPenaltyLastN = v; return this; }
        /** DRY restart sequences ({@code neuralfree.dry.sequence.breakers}); llama.cpp's defaults unless set. */
        public Builder drySequenceBreakers(List<String> v) {
            this.drySequenceBreakers = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(v, "sequence breakers")));
            return this;
        }
        /** top-n-sigma's n ({@code neuralfree.top.n.sigma}); 0 and below disables the stage. */
        public Builder topNSigma(float v) { this.topNSigma = v; return this; }
        /** XTC's trim probability ({@code neuralfree.xtc.probability}); 0 disables the stage. */
        public Builder xtcProbability(float v) { this.xtcProbability = v; return mark(Setting.XTC_PROBABILITY); }
        /** XTC's probability threshold ({@code neuralfree.xtc.threshold}); above 0.5 disables the stage. */
        public Builder xtcThreshold(float v) { this.xtcThreshold = v; return mark(Setting.XTC_THRESHOLD); }
        /** Mirostat version: {@link #MIROSTAT_OFF} (0), 1 or 2 ({@code neuralfree.mirostat}). */
        public Builder mirostat(int v) { this.mirostat = v; return mark(Setting.MIROSTAT); }
        /** Mirostat target entropy ({@code neuralfree.mirostat.tau}). */
        public Builder mirostatTau(float v) { this.mirostatTau = v; return mark(Setting.MIROSTAT_TAU); }
        /** Mirostat learning rate ({@code neuralfree.mirostat.eta}). */
        public Builder mirostatEta(float v) { this.mirostatEta = v; return mark(Setting.MIROSTAT_ETA); }
        /** Dynamic temperature half-width ({@code neuralfree.dynatemp.range}); 0 keeps the plain scaling. */
        public Builder dynatempRange(float v) { this.dynatempRange = v; return this; }
        /** Dynamic temperature entropy exponent ({@code neuralfree.dynatemp.exponent}). */
        public Builder dynatempExponent(float v) { this.dynatempExponent = v; return this; }
        /** adaptive-p's target probability ({@code neuralfree.adaptive.target}); negative disables the adaptation. */
        public Builder adaptiveTarget(float v) { this.adaptiveTarget = v; return this; }
        /** adaptive-p's EMA decay ({@code neuralfree.adaptive.decay}). */
        public Builder adaptiveDecay(float v) { this.adaptiveDecay = v; return this; }

        /** {@link #mirostat(int)} leaving {@link Setting#MIROSTAT} a fallback for a model's default. */
        public Builder fallbackMirostat(int v) { this.mirostat = v; return unmark(Setting.MIROSTAT); }
        /** {@link #mirostatTau(float)} leaving {@link Setting#MIROSTAT_TAU} a fallback. */
        public Builder fallbackMirostatTau(float v) { this.mirostatTau = v; return unmark(Setting.MIROSTAT_TAU); }
        /** {@link #mirostatEta(float)} leaving {@link Setting#MIROSTAT_ETA} a fallback. */
        public Builder fallbackMirostatEta(float v) { this.mirostatEta = v; return unmark(Setting.MIROSTAT_ETA); }
        /** {@link #xtcProbability(float)} leaving {@link Setting#XTC_PROBABILITY} a fallback. */
        public Builder fallbackXtcProbability(float v) { this.xtcProbability = v; return unmark(Setting.XTC_PROBABILITY); }
        /** {@link #xtcThreshold(float)} leaving {@link Setting#XTC_THRESHOLD} a fallback. */
        public Builder fallbackXtcThreshold(float v) { this.xtcThreshold = v; return unmark(Setting.XTC_THRESHOLD); }

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
            requireFinite("dryMultiplier", dryMultiplier);
            requireFinite("dryBase", dryBase);
            if (dryAllowedLength < 0) {
                throw new IllegalArgumentException(
                        PROP_DRY_ALLOWED_LENGTH + " must be >= 0, got " + dryAllowedLength);
            }
            if (dryPenaltyLastN < 0) {
                throw new IllegalArgumentException(
                        PROP_DRY_PENALTY_LAST_N + " must be >= 0 (0 = disabled), got " + dryPenaltyLastN);
            }
            requireFinite("topNSigma", topNSigma);
            requireUnit("xtcProbability", xtcProbability);
            requireUnit("xtcThreshold", xtcThreshold);
            if (mirostat < MIROSTAT_OFF || mirostat > 2) {
                throw new IllegalArgumentException(
                        PROP_MIROSTAT + " must be 0 (off), 1 or 2, got " + mirostat);
            }
            requireFinite("mirostatTau", mirostatTau);
            if (mirostatTau <= 0f) {
                throw new IllegalArgumentException(PROP_MIROSTAT_TAU + " must be > 0, got " + mirostatTau);
            }
            requireFinite("mirostatEta", mirostatEta);
            if (mirostatEta <= 0f) {
                throw new IllegalArgumentException(PROP_MIROSTAT_ETA + " must be > 0, got " + mirostatEta);
            }
            if (dynatempRange < 0f) {
                throw new IllegalArgumentException(PROP_DYNATEMP_RANGE + " must be >= 0 (0 = off), got " + dynatempRange);
            }
            requireFinite("dynatempExponent", dynatempExponent);
            if (dynatempExponent < 0f) {
                throw new IllegalArgumentException(
                        PROP_DYNATEMP_EXPONENT + " must be >= 0, got " + dynatempExponent);
            }
            requireFinite("adaptiveTarget", adaptiveTarget);
            requireFinite("adaptiveDecay", adaptiveDecay);
            if (!(adaptiveDecay >= 0f && adaptiveDecay <= 1f)) {
                throw new IllegalArgumentException(PROP_ADAPTIVE_DECAY + " must be in [0, 1], got " + adaptiveDecay);
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
