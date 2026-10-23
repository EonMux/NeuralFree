package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Speculative sampling of a target model with a cheaper draft model.
 *
 * <p><b>Models.</b> Both models map a {@code [1, L]} FLOAT32 tensor of token ids to logits whose last dimension is
 * the vocabulary; the output holds one row of logits per position ({@code L} rows), or fewer rows of which the last
 * predicts the next token. Each model runs on the backend of its parameters.</p>
 *
 * <p><b>Rounds.</b> The draft model proposes up to {@code draftSteps} tokens, each drawn from its next-token
 * distribution {@code q}. The target model then scores the proposals: with one row per position a single forward
 * pass gives every distribution {@code p}, otherwise each prefix is run separately. Proposal {@code x} is accepted
 * with probability {@code acceptThreshold * min(1, p(x) / q(x))}. The first rejected position is resampled from the
 * residual {@code p - acceptThreshold * min(p, q)}, normalized, which keeps every emitted token distributed exactly
 * as the target model's for any threshold. When all proposals are accepted, one more token is drawn from the target
 * distribution after them. A distribution is {@code softmax(logits / temperature)}; temperature 0 makes it one-hot
 * at the first maximal logit (greedy decoding).</p>
 *
 * <p><b>Device work.</b> Row selection, temperature scaling and the softmax (or argmax) run on the device. The
 * acceptance tests and the draws are Java decisions drawn from {@link Random}, so each distribution is read back as
 * its {@code V} probabilities, or as one index when greedy.</p>
 */
public final class SpeculativeDecoder implements AutoCloseable {
    private final Module draftModel;
    private final Module targetModel;
    private final SpeculativeDecoderConfig config;
    private final Random random;

    private volatile boolean closed = false;
    private long totalDraftTokens;
    private long totalAcceptedTokens;
    private long totalDecodingSteps;

    public SpeculativeDecoder(Module draftModel, Module targetModel, SpeculativeDecoderConfig config) {
        this(draftModel, targetModel, config, new Random());
    }

    /** A decoder whose acceptance tests and token draws come from {@code new Random(seed)}. */
    public SpeculativeDecoder(Module draftModel, Module targetModel, SpeculativeDecoderConfig config, long seed) {
        this(draftModel, targetModel, config, new Random(seed));
    }

    public SpeculativeDecoder(Module draftModel, Module targetModel) {
        this(draftModel, targetModel, SpeculativeDecoderConfig.defaults());
    }

    private SpeculativeDecoder(Module draftModel, Module targetModel, SpeculativeDecoderConfig config,
                               Random random) {
        Objects.requireNonNull(draftModel, "Draft model cannot be null");
        Objects.requireNonNull(targetModel, "Target model cannot be null");
        Objects.requireNonNull(config, "Config cannot be null");

        this.draftModel = draftModel;
        this.targetModel = targetModel;
        this.config = config;
        this.random = random;

        this.totalDraftTokens = 0;
        this.totalAcceptedTokens = 0;
        this.totalDecodingSteps = 0;
    }

    /**
     * {@code inputTokens} followed by exactly {@code maxNewTokens} generated tokens. Both models are put in
     * evaluation mode first, and every forward pass runs under {@link ComputationGraph#noGrad()}: decoding records
     * no autograd node.
     *
     * @throws IllegalArgumentException when {@code inputTokens} is empty or {@code maxNewTokens} is not positive
     * @throws IllegalStateException    when a model has no parameters (its backend is unknown), returns no logits,
     *                                  or yields a distribution without positive probability (NaN logits)
     */
    public float[] generate(float[] inputTokens, int maxNewTokens) {
        ensureOpen();
        Objects.requireNonNull(inputTokens, "Input tokens cannot be null");

        if (inputTokens.length == 0) {
            throw new IllegalArgumentException("Input tokens cannot be empty");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive: " + maxNewTokens);
        }

        ComputeBackend draftBackend = backendOf(draftModel, "Draft");
        ComputeBackend targetBackend = backendOf(targetModel, "Target");
        draftModel.eval();
        targetModel.eval();

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            float[] tokens = Arrays.copyOf(inputTokens, inputTokens.length + maxNewTokens);
            int length = inputTokens.length;
            int end = tokens.length;

            while (length < end) {
                int proposals = Math.min(config.draftSteps(), end - length);
                int prefix = length;

                float[][] draftDists = new float[proposals][];
                for (int i = 0; i < proposals; i++) {
                    draftDists[i] = lastRowDistribution(draftModel, "Draft", draftBackend, tokens, prefix + i);
                    tokens[prefix + i] = sample(draftDists[i], "draft");
                }
                float[][] targetDists = targetDistributions(targetBackend, tokens, prefix, proposals);

                int accepted = 0;
                while (accepted < proposals && accepts(targetDists[accepted], draftDists[accepted],
                        (int) tokens[prefix + accepted])) {
                    accepted++;
                }
                length = prefix + accepted;
                if (accepted < proposals) {
                    tokens[length++] = sampleResidual(targetDists[accepted], draftDists[accepted]);
                } else if (length < end) {
                    tokens[length++] = sample(targetDists[proposals], "target");
                }

                totalDraftTokens += proposals;
                totalAcceptedTokens += accepted;
                totalDecodingSteps++;
            }
            return tokens;
        }
    }

    /** Acceptance test of proposal {@code x}: a uniform draw below {@code acceptThreshold * min(1, p(x) / q(x))}. */
    private boolean accepts(float[] p, float[] q, int x) {
        float px = probabilityOf(p, x);
        float qx = probabilityOf(q, x);
        float ratio = qx > 0.0f ? Math.min(1.0f, px / qx) : 0.0f;
        return random.nextFloat() < config.acceptThreshold() * ratio;
    }

    /** A draw from {@code p - acceptThreshold * min(p, q)}, normalized; from {@code p} when that is all zero. */
    private int sampleResidual(float[] p, float[] q) {
        float threshold = config.acceptThreshold();
        float[] residual = new float[p.length];
        for (int x = 0; x < p.length; x++) {
            residual[x] = Math.max(0.0f, p[x] - threshold * Math.min(p[x], probabilityOf(q, x)));
        }
        Integer drawn = draw(residual);
        return drawn != null ? drawn : sample(p, "target");
    }

    private int sample(float[] distribution, String model) {
        Integer drawn = draw(distribution);
        if (drawn == null) {
            throw new IllegalStateException(model + " model distribution has no positive probability (NaN logits?)");
        }
        return drawn;
    }

    /**
     * The index whose running weight first exceeds {@code u * total} for a uniform {@code u}, among the indices of
     * positive weight; the last positive index when rounding leaves none, null when no weight is positive.
     */
    private Integer draw(float[] weights) {
        double total = 0.0;
        int last = -1;
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] > 0.0f) {
                total += weights[i];
                last = i;
            }
        }
        if (last < 0) {
            return null;
        }
        double target = random.nextDouble() * total;
        double cumulative = 0.0;
        for (int i = 0; i < last; i++) {
            if (weights[i] > 0.0f) {
                cumulative += weights[i];
                if (target < cumulative) {
                    return i;
                }
            }
        }
        return last;
    }

    private static float probabilityOf(float[] distribution, int x) {
        return x >= 0 && x < distribution.length ? distribution[x] : 0.0f;
    }

    /**
     * The target distributions for the {@code proposals} positions after {@code prefix} and the position after them
     * ({@code proposals + 1} rows): from one forward pass when the target emits one row per position, otherwise from
     * one pass per prefix.
     */
    private float[][] targetDistributions(ComputeBackend backend, float[] tokens, int prefix, int proposals) {
        int length = prefix + proposals;
        float[][] dists = new float[proposals + 1][];
        try (Tensor input = tokenRow(backend, tokens, length);
             Tensor logits = targetModel.forward(input);
             Tensor rows = logitRows(logits, "Target")) {
            int rowCount = rows.shape().dim(0);
            if (rowCount == length) {
                return distributions(rows, prefix - 1, proposals + 1);
            }
            dists[proposals] = distributions(rows, rowCount - 1, 1)[0];
        }
        for (int i = 0; i < proposals; i++) {
            dists[i] = lastRowDistribution(targetModel, "Target", backend, tokens, prefix + i);
        }
        return dists;
    }

    /** The next-token distribution after the first {@code length} tokens: the model's last row of logits. */
    private float[] lastRowDistribution(Module model, String name, ComputeBackend backend, float[] tokens,
                                        int length) {
        try (Tensor input = tokenRow(backend, tokens, length);
             Tensor logits = model.forward(input);
             Tensor rows = logitRows(logits, name)) {
            return distributions(rows, rows.shape().dim(0) - 1, 1)[0];
        }
    }

    /** {@code count} distributions from rows {@code [first, first + count)} of a {@code [rows, V]} logit matrix. */
    private float[][] distributions(Tensor rows, int first, int count) {
        int vocab = rows.shape().dim(1);
        float[][] out = new float[count][];
        try (Tensor picked = rows.slice(0, first, (long) first + count)) {
            if (config.temperature() == 0.0f) {
                try (Tensor best = TensorOps.argmax(picked, 1)) {
                    int[] index = best.toIntArray();
                    for (int r = 0; r < count; r++) {
                        out[r] = new float[vocab];
                        out[r][index[r]] = 1.0f;
                    }
                }
                return out;
            }
            try (Tensor divisor = Tensor.scalar(picked.backend(), config.temperature());
                 Tensor scaled = TensorOps.div(picked, divisor);
                 Tensor probs = TensorOps.softmax(scaled, 1)) {
                float[] flat = probs.toFloatArray();
                for (int r = 0; r < count; r++) {
                    out[r] = Arrays.copyOfRange(flat, r * vocab, (r + 1) * vocab);
                }
                return out;
            }
        }
    }

    /** {@code logits} as a {@code [rows, V]} matrix with {@code V} its last dimension. */
    private static Tensor logitRows(Tensor logits, String model) {
        Shape shape = logits.shape();
        int vocab = shape.rank() == 0 ? 0 : shape.dim(shape.rank() - 1);
        if (vocab == 0 || logits.numElements() == 0) {
            throw new IllegalStateException(model + " model returned no logits: shape " + shape);
        }
        return logits.reshape(Shape.matrix((int) (logits.numElements() / vocab), vocab));
    }

    private static Tensor tokenRow(ComputeBackend backend, float[] tokens, int length) {
        return Tensor.fromArray(backend, new float[][]{Arrays.copyOf(tokens, length)});
    }

    private static ComputeBackend backendOf(Module model, String name) {
        ComputeBackend backend = model.backend();
        if (backend == null) {
            throw new IllegalStateException(name + " model has no parameters, so its backend is unknown");
        }
        return backend;
    }

    public double acceptanceRate() {
        return totalDraftTokens > 0 ? (double) totalAcceptedTokens / totalDraftTokens : 0.0;
    }

    public long totalDraftTokens() {
        return totalDraftTokens;
    }

    public long totalAcceptedTokens() {
        return totalAcceptedTokens;
    }

    public long totalDecodingSteps() {
        return totalDecodingSteps;
    }

    public SpeculativeDecoderConfig config() {
        return config;
    }

    public Module draftModel() {
        return draftModel;
    }

    public Module targetModel() {
        return targetModel;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SpeculativeDecoder has been closed");
        }
    }

    /** Marks the decoder closed; the models belong to the caller and stay open. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "SpeculativeDecoder{" +
            "draftModel=" + draftModel.moduleName() +
            ", targetModel=" + targetModel.moduleName() +
            ", acceptanceRate=" + String.format("%.2f%%", acceptanceRate() * 100) +
            ", closed=" + closed +
            '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Module draftModel;
        private Module targetModel;
        private SpeculativeDecoderConfig config = SpeculativeDecoderConfig.defaults();
        private Long seed;

        public Builder draftModel(Module draftModel) {
            this.draftModel = draftModel;
            return this;
        }

        public Builder targetModel(Module targetModel) {
            this.targetModel = targetModel;
            return this;
        }

        public Builder config(SpeculativeDecoderConfig config) {
            this.config = config;
            return this;
        }

        public Builder draftSteps(int steps) {
            this.config = new SpeculativeDecoderConfig(
                steps, config.acceptThreshold(), config.temperature()
            );
            return this;
        }

        /** Seeds the acceptance tests and token draws; without it they use an unseeded {@link Random}. */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public SpeculativeDecoder build() {
            Objects.requireNonNull(draftModel, "Draft model cannot be null");
            Objects.requireNonNull(targetModel, "Target model cannot be null");
            return seed != null
                ? new SpeculativeDecoder(draftModel, targetModel, config, seed.longValue())
                : new SpeculativeDecoder(draftModel, targetModel, config);
        }
    }
}
