package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The sampling defaults a model carries: the {@code general.sampling.*} keys llama.cpp's converter writes from
 * {@code generation_config.json} (gguf-py/gguf/metadata.py:71-86) and the {@code neuralfree.sampling.*} keys for the
 * generation_config values llama.cpp has no key for. Every value is absent (null) unless the model gives it. Generation
 * applies a present value only to a setting the caller left unset ({@link SamplerConfig#withModelDefaults},
 * {@link GenerationRequest#withModelDefaults}), as llama.cpp does ({@code common_init_sampler_from_model},
 * common/common.cpp:1088-1147) and as vLLM does with {@code generation_config}. Immutable.
 *
 * <p>{@code general.sampling.sequence} is the configurable chain order ({@link StageKind}): a
 * '{@code ;}'-separated list of llama.cpp stage names is read; anything else (an abbreviated or free-form value,
 * as {@code generation_config} hands this metadata through) is logged as ignored, like the llama.cpp keys
 * {@code general.sampling.{mirostat, mirostat_tau, mirostat_eta, xtc_probability, xtc_threshold}}.
 */
public final class SamplingDefaults {

    private static final Logger LOGGER = LogManager.getLogger(SamplingDefaults.class);

    /** Keys of llama.cpp sampler settings without a stage in NeuralFree's chain. */
    private static final String[] UNSUPPORTED_KEYS = {
            GgufKeys.GENERAL_SAMPLING_MIROSTAT,
            GgufKeys.GENERAL_SAMPLING_MIROSTAT_TAU,
            GgufKeys.GENERAL_SAMPLING_MIROSTAT_ETA,
            GgufKeys.GENERAL_SAMPLING_XTC_PROBABILITY,
            GgufKeys.GENERAL_SAMPLING_XTC_THRESHOLD,
    };

    /** No model defaults. */
    public static final SamplingDefaults NONE = builder().build();

    private final Float temp;
    private final Integer topK;
    private final Float topP;
    private final Float minP;
    private final Float typicalP;
    private final Float penaltyRepeat;
    private final Integer penaltyLastN;
    private final Float presencePenalty;
    private final Float frequencyPenalty;
    private final Integer maxNewTokens;
    private final List<StageKind> sequence;

    private SamplingDefaults(Builder b) {
        this.temp = b.temp;
        this.topK = b.topK;
        this.topP = b.topP;
        this.minP = b.minP;
        this.typicalP = b.typicalP;
        this.penaltyRepeat = b.penaltyRepeat;
        this.penaltyLastN = b.penaltyLastN;
        this.presencePenalty = b.presencePenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.maxNewTokens = b.maxNewTokens;
        this.sequence = b.sequence;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * The defaults in a model's metadata; only present keys are read. Integer settings take any integer type in the
     * int range, the others any numeric type.
     *
     * @throws GgufFormatException naming the key when a present value has the wrong type, is not finite or is out of
     *                             the range {@link SamplerConfig} accepts
     */
    public static SamplingDefaults read(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        Builder b = new Builder();
        b.temp = readFloat(kv, GgufKeys.GENERAL_SAMPLING_TEMP, Range.FINITE);
        b.topK = readInt(kv, GgufKeys.GENERAL_SAMPLING_TOP_K, Range.FINITE);
        b.topP = readFloat(kv, GgufKeys.GENERAL_SAMPLING_TOP_P, Range.UNIT);
        b.minP = readFloat(kv, GgufKeys.GENERAL_SAMPLING_MIN_P, Range.UNIT);
        b.penaltyRepeat = readFloat(kv, GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, Range.POSITIVE);
        b.penaltyLastN = readInt(kv, GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, Range.PENALTY_WINDOW);
        b.presencePenalty = readFloat(kv, GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, Range.FINITE);
        b.frequencyPenalty = readFloat(kv, GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY, Range.FINITE);
        b.maxNewTokens = readInt(kv, GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, Range.TOKEN_BUDGET);
        b.sequence = readSequence(kv);
        logUnsupported(kv);
        SamplingDefaults d = new SamplingDefaults(b);
        if (d.isEmpty()) {
            return NONE;
        }
        LOGGER.info("model sampling defaults {} fill the settings a request leaves unset ({}=false turns them off)",
                d, EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS);
        return d;
    }

    /**
     * The defaults generation applies for {@code model} under {@code config}: the model's own, or {@link #NONE} when
     * {@link EngineConfig#modelSamplingDefaults()} is off.
     */
    public static SamplingDefaults forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (!config.modelSamplingDefaults()) {
            return NONE;
        }
        return Objects.requireNonNull(model.modelSamplingDefaults(), "modelSamplingDefaults");
    }

    /** Temperature ({@code general.sampling.temp}), or null. */
    public Float temp() { return temp; }
    /** Top-k ({@code general.sampling.top_k}), or null. */
    public Integer topK() { return topK; }
    /** Top-p ({@code general.sampling.top_p}), or null. */
    public Float topP() { return topP; }
    /** Min-p ({@code general.sampling.min_p}), or null. */
    public Float minP() { return minP; }
    /** Typical-p, or null; no metadata key carries it, so only {@link #builder()} sets it. */
    public Float typicalP() { return typicalP; }
    /** Repetition penalty ({@code general.sampling.penalty_repeat}), or null. */
    public Float penaltyRepeat() { return penaltyRepeat; }
    /** Penalty window ({@code general.sampling.penalty_last_n}, -1 = whole context), or null. */
    public Integer penaltyLastN() { return penaltyLastN; }
    /** Presence penalty ({@code neuralfree.sampling.presence_penalty}), or null. */
    public Float presencePenalty() { return presencePenalty; }
    /** Frequency penalty ({@code neuralfree.sampling.frequency_penalty}), or null. */
    public Float frequencyPenalty() { return frequencyPenalty; }
    /** New-token budget ({@code neuralfree.sampling.max_new_tokens}, -1 = unlimited), or null. */
    public Integer maxNewTokens() { return maxNewTokens; }
    /** The configurable chain order ({@code general.sampling.sequence}, unmodifiable), or null. */
    public List<StageKind> sequence() { return sequence; }

    /** True when the model gives no default. */
    public boolean isEmpty() {
        return temp == null && topK == null && topP == null && minP == null && typicalP == null
                && penaltyRepeat == null && penaltyLastN == null && presencePenalty == null
                && frequencyPenalty == null && maxNewTokens == null && sequence == null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SamplingDefaults)) {
            return false;
        }
        SamplingDefaults d = (SamplingDefaults) o;
        return Objects.equals(temp, d.temp) && Objects.equals(topK, d.topK) && Objects.equals(topP, d.topP)
                && Objects.equals(minP, d.minP) && Objects.equals(typicalP, d.typicalP)
                && Objects.equals(penaltyRepeat, d.penaltyRepeat) && Objects.equals(penaltyLastN, d.penaltyLastN)
                && Objects.equals(presencePenalty, d.presencePenalty)
                && Objects.equals(frequencyPenalty, d.frequencyPenalty) && Objects.equals(maxNewTokens, d.maxNewTokens)
                && Objects.equals(sequence, d.sequence);
    }

    @Override
    public int hashCode() {
        return Objects.hash(temp, topK, topP, minP, typicalP, penaltyRepeat, penaltyLastN, presencePenalty,
                frequencyPenalty, maxNewTokens, sequence);
    }

    @Override
    public String toString() {
        if (isEmpty()) {
            return "SamplingDefaults{none}";
        }
        List<String> parts = new ArrayList<>();
        add(parts, "temp", temp);
        add(parts, "topK", topK);
        add(parts, "topP", topP);
        add(parts, "minP", minP);
        add(parts, "typicalP", typicalP);
        add(parts, "repeat", penaltyRepeat);
        add(parts, "lastN", penaltyLastN);
        add(parts, "presence", presencePenalty);
        add(parts, "frequency", frequencyPenalty);
        add(parts, "maxNewTokens", maxNewTokens);
        if (sequence != null) {
            List<String> labels = new ArrayList<>();
            for (StageKind kind : sequence) {
                labels.add(kind.label());
            }
            parts.add("sequence=" + String.join(";", labels));
        }
        return "SamplingDefaults{" + String.join(", ", parts) + '}';
    }

    private static void add(List<String> parts, String name, Number value) {
        if (value != null) {
            parts.add(name + "=" + value);
        }
    }

    private static Float readFloat(GgufKv kv, String key, Range range) {
        if (!kv.has(key)) {
            return null;
        }
        float v = kv.f32(key, 0f);
        String problem = range.violation(v);
        if (problem != null) {
            throw new GgufFormatException("metadata key '" + key + "' value " + v + " " + problem);
        }
        return v;
    }

    private static Integer readInt(GgufKv kv, String key, Range range) {
        if (!kv.has(key)) {
            return null;
        }
        int v = kv.i32(key, 0);
        String problem = range.violation(v);
        if (problem != null) {
            throw new GgufFormatException("metadata key '" + key + "' value " + v + " " + problem);
        }
        return v;
    }

    private static void logUnsupported(GgufKv kv) {
        List<String> present = new ArrayList<>();
        for (String key : UNSUPPORTED_KEYS) {
            if (kv.has(key)) {
                present.add(key);
            }
        }
        if (!present.isEmpty()) {
            LOGGER.info("model sampling defaults {} are ignored: NeuralFree's sampler chain has no such stage",
                    present);
        }
    }

    /**
     * {@code general.sampling.sequence} as a configurable chain order ({@link StageKind}), or null when the key is
     * absent or its value is not a '{@code ;}'-separated list of llama.cpp stage names (a free-form
     * {@code generation_config} value: logged as with the keys without a stage, never applied).
     */
    private static List<StageKind> readSequence(GgufKv kv) {
        if (!kv.has(GgufKeys.GENERAL_SAMPLING_SEQUENCE)) {
            return null;
        }
        String joined = kv.str(GgufKeys.GENERAL_SAMPLING_SEQUENCE, null);
        if (joined == null) {
            return null;
        }
        try {
            return StageKind.parseOrder(joined);
        } catch (IllegalArgumentException e) {
            LOGGER.info("model sampling default general.sampling.sequence={} is ignored: {}",
                    joined, e.getMessage());
            return null;
        }
    }

    /** The values {@link SamplerConfig.Builder#build()} and {@link GenerationRequest.Builder#build()} accept. */
    private enum Range {
        FINITE(null),
        UNIT("must be in [0, 1]"),
        POSITIVE("must be > 0"),
        PENALTY_WINDOW("must be >= -1 (-1 = whole context, 0 = disabled)"),
        TOKEN_BUDGET("must be positive, or -1 for unlimited");

        private final String rule;

        Range(String rule) {
            this.rule = rule;
        }

        /** Why {@code v} is rejected, or null when it is accepted. */
        String violation(double v) {
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                return "must be finite";
            }
            boolean ok;
            switch (this) {
                case UNIT: ok = v >= 0 && v <= 1; break;
                case POSITIVE: ok = v > 0; break;
                case PENALTY_WINDOW: ok = v >= SamplerConfig.PENALTY_WHOLE_CONTEXT; break;
                case TOKEN_BUDGET: ok = v > 0 || v == GenerationRequest.UNLIMITED; break;
                default: ok = true; break;
            }
            return ok ? null : rule;
        }
    }

    /** Builds defaults programmatically; unset values stay absent. */
    public static final class Builder {
        private Float temp;
        private Integer topK;
        private Float topP;
        private Float minP;
        private Float typicalP;
        private Float penaltyRepeat;
        private Integer penaltyLastN;
        private Float presencePenalty;
        private Float frequencyPenalty;
        private Integer maxNewTokens;
        private List<StageKind> sequence;

        private Builder() {
        }

        public Builder temp(float v) { this.temp = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(float v) { this.topP = v; return this; }
        public Builder minP(float v) { this.minP = v; return this; }
        public Builder typicalP(float v) { this.typicalP = v; return this; }
        public Builder penaltyRepeat(float v) { this.penaltyRepeat = v; return this; }
        public Builder penaltyLastN(int v) { this.penaltyLastN = v; return this; }
        public Builder presencePenalty(float v) { this.presencePenalty = v; return this; }
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        /** The configurable chain order, as llama.cpp's stage names ({@code general.sampling.sequence}). */
        public Builder sequence(String joined) {
            this.sequence = StageKind.parseOrder(joined);
            return this;
        }

        /** The configurable chain order. */
        public Builder sequence(List<StageKind> order) {
            List<StageKind> parse = new ArrayList<>();
            for (StageKind kind : order) {
                parse.add(StageKind.ofLabel(kind.label()));
            }
            this.sequence = Collections.unmodifiableList(parse);
            return this;
        }

        /** @throws IllegalArgumentException naming the setting when a value is out of range */
        public SamplingDefaults build() {
            check("temp", temp, Range.FINITE);
            check("topP", topP, Range.UNIT);
            check("minP", minP, Range.UNIT);
            check("typicalP", typicalP, Range.UNIT);
            check("penaltyRepeat", penaltyRepeat, Range.POSITIVE);
            check("penaltyLastN", penaltyLastN, Range.PENALTY_WINDOW);
            check("presencePenalty", presencePenalty, Range.FINITE);
            check("frequencyPenalty", frequencyPenalty, Range.FINITE);
            check("maxNewTokens", maxNewTokens, Range.TOKEN_BUDGET);
            return new SamplingDefaults(this);
        }

        private static void check(String name, Number v, Range range) {
            if (v == null) {
                return;
            }
            String problem = range.violation(v.doubleValue());
            if (problem != null) {
                throw new IllegalArgumentException(name + " " + problem + ", got " + v);
            }
        }
    }
}
