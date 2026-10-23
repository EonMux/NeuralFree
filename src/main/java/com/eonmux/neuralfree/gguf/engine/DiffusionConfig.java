package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;

/**
 * Masked-diffusion generation settings (dream, llada, llada-moe, rnd1), mirroring llama.cpp
 * {@code diffusion_params} (llama.cpp examples/diffusion/diffusion.h) and the CLI defaults of
 * {@code common_params_diffusion} (llama.cpp common/common.h). Immutable.
 *
 * <p>The canvas is the prompt followed by mask tokens up to {@link #maxLength()} tokens in total. Each step runs
 * a non-causal forward over the canvas, samples every masked row through the chain top-k, top-p, temperature,
 * seeded draw (llama.cpp diffusion.cpp sampler setup), scores it by {@link Algorithm}, and unmasks the number of rows
 * the {@link Schedule} transfers.
 */
public final class DiffusionConfig {

    /** Confidence measure used to pick the rows to unmask (llama.cpp {@code diffusion_algorithm}). */
    public enum Algorithm {
        /** Unmask each masked row with probability {@code transfer / masked}. */
        ORIGIN(0),
        /** Negative entropy of the sampled distribution. */
        ENTROPY_BASED(1),
        /** Probability gap between the two best candidates. */
        MARGIN_BASED(2),
        /** Uniform random confidence. */
        RANDOM(3),
        /** Probability of the sampled token. */
        CONFIDENCE_BASED(4);

        private final int code;

        Algorithm(int code) {
            this.code = code;
        }

        /** llama.cpp enum value ({@code --diffusion-algorithm}). */
        public int code() {
            return code;
        }

        /**
         * @throws IllegalArgumentException for an unknown code
         */
        public static Algorithm fromCode(int code) {
            for (Algorithm a : values()) {
                if (a.code == code) {
                    return a;
                }
            }
            throw new IllegalArgumentException("unknown diffusion algorithm " + code + "; use 0-4");
        }
    }

    /** How many rows each step unmasks (llama.cpp {@code diffusion_transfer_schedule}). */
    public enum Schedule {
        /** Dream style: {@code remaining * (1 - s/t)} with {@code t, s} from the step and {@link #eps()}. */
        TIMESTEP,
        /** LLaDA style: blocks of {@link #blockLength()} tokens, each unmasked evenly over its share of steps. */
        BLOCK
    }

    /** Whether row {@code i} is predicted from logits row {@code i - 1}. */
    public enum ShiftLogits {
        /** Use {@code diffusion.shift_logits} metadata; true when the key is absent (llama.cpp diffusion-cli.cpp). */
        FROM_MODEL,
        /** Always shift. */
        SHIFT,
        /** Never shift. */
        NO_SHIFT
    }

    /** {@link #maskTokenId()} value meaning "the tokenizer's mask token". */
    public static final int MASK_FROM_TOKENIZER = -1;
    /** {@link #maxLength()} value meaning "the memory's ubatch size" (llama.cpp uses {@code n_ubatch}). */
    public static final int MAX_LENGTH_FROM_UBATCH = 0;

    private final int steps;
    private final Algorithm algorithm;
    private final Schedule schedule;
    private final float eps;
    private final int blockLength;
    private final float algTemp;
    private final float cfgScale;
    private final boolean addGumbelNoise;
    private final ShiftLogits shiftLogits;
    private final int maskTokenId;
    private final int maxLength;
    private final float temperature;
    private final int topK;
    private final float topP;
    private final long seed;

    private DiffusionConfig(Builder b) {
        this.steps = b.steps;
        this.algorithm = b.algorithm;
        this.schedule = b.schedule;
        this.eps = b.eps;
        this.blockLength = b.blockLength;
        this.algTemp = b.algTemp;
        this.cfgScale = b.cfgScale;
        this.addGumbelNoise = b.addGumbelNoise;
        this.shiftLogits = b.shiftLogits;
        this.maskTokenId = b.maskTokenId;
        this.maxLength = b.maxLength;
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.seed = b.seed;
    }

    /** Defaults: 128 steps, confidence-based, timestep schedule with eps 0.001, llama.cpp sampling defaults. */
    public static DiffusionConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public int steps() { return steps; }
    public Algorithm algorithm() { return algorithm; }
    public Schedule schedule() { return schedule; }
    /** Timestep epsilon; 0 for the block schedule. */
    public float eps() { return eps; }
    /** Block length; 0 for the timestep schedule. */
    public int blockLength() { return blockLength; }
    /** Temperature of the confidence draw; 0 picks the most confident rows deterministically. */
    public float algTemp() { return algTemp; }
    /** Classifier-free guidance scale; 0 disables it. */
    public float cfgScale() { return cfgScale; }
    /** Whether Gumbel noise is added to the logits when the temperature is positive. */
    public boolean addGumbelNoise() { return addGumbelNoise; }
    public ShiftLogits shiftLogits() { return shiftLogits; }
    /** Mask token id, or {@link #MASK_FROM_TOKENIZER}. */
    public int maskTokenId() { return maskTokenId; }
    /** Total canvas length (prompt + generated), or {@link #MAX_LENGTH_FROM_UBATCH}. */
    public int maxLength() { return maxLength; }
    /** Sampling temperature; {@code <= 0} disables the temperature stage. */
    public float temperature() { return temperature; }
    /** Top-k; {@code <= 0} disables it. */
    public int topK() { return topK; }
    /** Top-p; 1 disables it. */
    public float topP() { return topP; }
    /** RNG seed, or {@link SamplerConfig#RANDOM_SEED}. */
    public long seed() { return seed; }

    /** Classifier-free guidance is active. */
    public boolean cfgEnabled() {
        return cfgScale > 0f;
    }

    /**
     * Resolves {@link #shiftLogits()} against the model metadata.
     *
     * @param metadataShift value of {@code diffusion.shift_logits}, or null when the key is absent
     */
    public boolean resolveShiftLogits(Boolean metadataShift) {
        switch (shiftLogits) {
            case SHIFT:
                return true;
            case NO_SHIFT:
                return false;
            default:
                return metadataShift == null || metadataShift;
        }
    }

    /**
     * Resolves {@link #maxLength()} and checks it against the prompt.
     *
     * @param ubatch memory ubatch, used for {@link #MAX_LENGTH_FROM_UBATCH}
     * @throws IllegalArgumentException when the canvas leaves no room after the prompt
     */
    public int resolveMaxLength(int promptLength, int ubatch) {
        int len = maxLength == MAX_LENGTH_FROM_UBATCH ? ubatch : maxLength;
        if (len <= promptLength) {
            throw new IllegalArgumentException("diffusion canvas of " + len + " tokens leaves no room after a "
                    + promptLength + "-token prompt; raise the max length (or -Dneuralfree.ubatch)");
        }
        return len;
    }

    /**
     * Steps per block for a resolved canvas length: {@link #steps()} for the timestep schedule; for the block
     * schedule the canvas is split into {@code maxLength / blockLength} blocks that share the steps evenly
     * (llama.cpp diffusion.cpp asserts both divisions).
     *
     * @throws IllegalArgumentException when the canvas or the steps do not divide evenly
     */
    public int stepsPerBlock(int resolvedMaxLength) {
        if (schedule == Schedule.TIMESTEP) {
            return steps;
        }
        if (resolvedMaxLength % blockLength != 0) {
            throw new IllegalArgumentException("diffusion max length " + resolvedMaxLength
                    + " is not a multiple of the block length " + blockLength);
        }
        int blocks = resolvedMaxLength / blockLength;
        if (steps % blocks != 0) {
            throw new IllegalArgumentException("diffusion steps " + steps + " are not a multiple of the "
                    + blocks + " blocks of length " + blockLength);
        }
        return steps / blocks;
    }

    public Builder toBuilder() {
        Builder b = builder().steps(steps).algorithm(algorithm).algTemp(algTemp).cfgScale(cfgScale)
                .addGumbelNoise(addGumbelNoise).shiftLogits(shiftLogits).maskTokenId(maskTokenId)
                .maxLength(maxLength).temperature(temperature).topK(topK).topP(topP).seed(seed);
        return schedule == Schedule.TIMESTEP ? b.timestepSchedule(eps) : b.blockSchedule(blockLength);
    }

    @Override
    public String toString() {
        return "DiffusionConfig{steps=" + steps + ", algorithm=" + algorithm + ", schedule=" + schedule
                + (schedule == Schedule.TIMESTEP ? ", eps=" + eps : ", blockLength=" + blockLength)
                + ", algTemp=" + algTemp + ", cfg=" + cfgScale + ", gumbel=" + addGumbelNoise + ", shift="
                + shiftLogits + ", mask=" + (maskTokenId == MASK_FROM_TOKENIZER ? "tokenizer" : maskTokenId)
                + ", maxLength=" + (maxLength == MAX_LENGTH_FROM_UBATCH ? "ubatch" : String.valueOf(maxLength))
                + ", temp=" + temperature + ", topK=" + topK + ", topP=" + topP + ", seed="
                + (seed == SamplerConfig.RANDOM_SEED ? "random" : String.valueOf(seed)) + '}';
    }

    public static final class Builder {
        private int steps = 128;
        private Algorithm algorithm = Algorithm.CONFIDENCE_BASED;
        private Schedule schedule = Schedule.TIMESTEP;
        private float eps = 1e-3f;
        private int blockLength;
        private float algTemp;
        private float cfgScale;
        private boolean addGumbelNoise;
        private ShiftLogits shiftLogits = ShiftLogits.FROM_MODEL;
        private int maskTokenId = MASK_FROM_TOKENIZER;
        private int maxLength = MAX_LENGTH_FROM_UBATCH;
        private float temperature = 0.8f;
        private int topK = 40;
        private float topP = 0.95f;
        private long seed = SamplerConfig.RANDOM_SEED;

        private Builder() {
        }

        public Builder steps(int v) { this.steps = v; return this; }
        public Builder algorithm(Algorithm v) { this.algorithm = Objects.requireNonNull(v, "algorithm"); return this; }

        /** Dream-style timestep schedule with epsilon {@code eps} in (0, 1). */
        public Builder timestepSchedule(float eps) {
            this.schedule = Schedule.TIMESTEP;
            this.eps = eps;
            this.blockLength = 0;
            return this;
        }

        /** LLaDA-style block schedule with blocks of {@code blockLength} tokens. */
        public Builder blockSchedule(int blockLength) {
            this.schedule = Schedule.BLOCK;
            this.blockLength = blockLength;
            this.eps = 0f;
            return this;
        }

        public Builder algTemp(float v) { this.algTemp = v; return this; }
        public Builder cfgScale(float v) { this.cfgScale = v; return this; }
        public Builder addGumbelNoise(boolean v) { this.addGumbelNoise = v; return this; }
        public Builder shiftLogits(ShiftLogits v) { this.shiftLogits = Objects.requireNonNull(v, "shiftLogits"); return this; }
        public Builder maskTokenId(int v) { this.maskTokenId = v; return this; }
        public Builder maxLength(int v) { this.maxLength = v; return this; }
        public Builder temperature(float v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(float v) { this.topP = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }

        /** @throws IllegalArgumentException naming the setting when a value is out of range */
        public DiffusionConfig build() {
            if (steps <= 0) {
                throw new IllegalArgumentException("diffusion steps must be positive, got " + steps);
            }
            if (schedule == Schedule.TIMESTEP && !(eps > 0f && eps < 1f)) {
                throw new IllegalArgumentException("diffusion eps must be in (0, 1) for the timestep schedule, got " + eps);
            }
            if (schedule == Schedule.BLOCK && blockLength <= 0) {
                throw new IllegalArgumentException("diffusion block length must be positive, got " + blockLength);
            }
            requireNonNegativeFinite("algTemp", algTemp);
            requireNonNegativeFinite("cfgScale", cfgScale);
            if (Float.isNaN(temperature) || Float.isInfinite(temperature)) {
                throw new IllegalArgumentException("diffusion temperature must be finite, got " + temperature);
            }
            if (!(topP >= 0f && topP <= 1f)) {
                throw new IllegalArgumentException("diffusion topP must be in [0, 1], got " + topP);
            }
            if (maskTokenId < MASK_FROM_TOKENIZER) {
                throw new IllegalArgumentException("mask token id must be >= 0, or -1 for the tokenizer's, got " + maskTokenId);
            }
            if (maxLength < 0) {
                throw new IllegalArgumentException("diffusion max length must be >= 0 (0 = ubatch), got " + maxLength);
            }
            if (seed < SamplerConfig.RANDOM_SEED) {
                throw new IllegalArgumentException("diffusion seed must be >= 0, or -1 for a random seed, got " + seed);
            }
            DiffusionConfig config = new DiffusionConfig(this);
            if (maxLength != MAX_LENGTH_FROM_UBATCH) {
                config.stepsPerBlock(maxLength);
            }
            return config;
        }

        private static void requireNonNegativeFinite(String name, float v) {
            if (!(v >= 0f) || Float.isInfinite(v)) {
                throw new IllegalArgumentException("diffusion " + name + " must be finite and >= 0, got " + v);
            }
        }
    }
}
