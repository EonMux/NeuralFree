package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;

import java.util.Objects;

/**
 * Model output after the last layer, resolved by {@code HeadResolver}.
 * <ol>
 *   <li>hyper-connection head ({@code output_hc_*}) collapses the {@code hcMult} streams when {@link #hcHead()};</li>
 *   <li>{@code output_norm} (+ bias) when present, over {@link #outputNormGroups()} equal slices of the row
 *       (k2-horizon's grouped RMS norm, models/k2-horizon.cpp:124-135, 352);</li>
 *   <li>{@link #embeddingProjection()} ({@code output.weight} of gemma-embedding2, {@code {n_embd, n_embd_out}})
 *       maps the normed rows to the {@link #hiddenWidth(int)} embedding width before pooling
 *       (models/gemma-embedding2.cpp:41, 206-212); such a model has no LM head;</li>
 *   <li>{@link #embeddingOutput()} ({@code cls.output} of qwen35, {@code {n_embd, n_embd_out}}) with
 *       {@link #embeddingOutputBias()} maps the normed rows to the {@link #embeddingWidth(int)} embedding rows
 *       ({@code t_embd}) that pooling reads, while the LM head keeps reading the normed rows
 *       (models/qwen35.cpp:47-48, 222-229);</li>
 *   <li>generation: {@code h = project_out · h} when {@link #outputProjection()} is present (OPT's
 *       {@code word_embed_proj_dim != hidden_size}); then {@code logits = output · h} (+ {@code output.bias});
 *       {@code output} is {@code token_embd} when {@link #tied()}; then {@code logits = logits / logitScale} or
 *       {@code * logitScale} per
 *       {@link #logitScaleDivides()} (granite divides, cohere / grok multiply); then the final softcap
 *       {@code cap * tanh(logits / cap)}; logits {@code [maskedLogitsStart, maskedLogitsEnd)} are then set to
 *       {@code -Float.MAX_VALUE} (chameleon's image tokens); a drafter with a reduced vocabulary (eagle3, the
 *       FR-Spec MTP draft head) maps draft ids through {@code d2t}. The logits rows have
 *       {@link #vocabRows()} entries; qwen3tts's {@code -inf} prefix up to the {@code token_embd} row count
 *       (qwen3vl.cpp:175-191) is applied around the backend by {@code ops.audio.TtsBackbone.qwen3tts};</li>
 *   <li>embedding / classifier models: pooling by {@link #pooling()}, optional {@code dense_2} / {@code dense_3}
 *       projections, and the {@code cls} / {@code cls.output} classifier head.</li>
 * </ol>
 * Immutable.
 */
public final class HeadSpec {

    private final TensorRef outputNorm;
    private final TensorRef outputNormBias;
    private final TensorRef outputProjection;
    private final TensorRef output;
    private final boolean tied;
    private final TensorRef outputBias;
    private final float finalSoftcap;
    private final float logitScale;
    private final boolean logitScaleDivides;
    private final TensorRef logitScaleVector;
    private final float sigmoidCapScale;
    private final float sigmoidCapShift;
    private final float sigmoidCapTemperature;
    private final boolean hcHead;
    private final TensorRef d2t;
    private final PoolingKind pooling;
    private final TensorRef dense2;
    private final TensorRef dense3;
    private final TensorRef cls;
    private final TensorRef clsOutput;
    private final TensorRef clsNorm;
    private final TensorRef colbertOutput;
    private final TensorRef colbertOutputBias;
    private final TensorRef sparseOutput;
    private final TensorRef sparseOutputBias;
    private final ClassifierScope classifierScope;
    private final TensorRef embeddingProjection;
    private final TensorRef embeddingOutput;
    private final TensorRef embeddingOutputBias;
    private final int maskedLogitsStart;
    private final int maskedLogitsEnd;
    private final int outputNormGroups;

    private HeadSpec(Builder b) {
        this.outputNorm = b.outputNorm;
        this.outputNormBias = b.outputNormBias;
        this.outputProjection = b.outputProjection;
        this.output = b.output;
        this.tied = b.tied;
        this.outputBias = b.outputBias;
        this.finalSoftcap = b.finalSoftcap;
        this.logitScale = b.logitScale;
        this.logitScaleDivides = b.logitScaleDivides;
        this.logitScaleVector = b.logitScaleVector;
        this.sigmoidCapScale = b.sigmoidCapScale;
        this.sigmoidCapShift = b.sigmoidCapShift;
        this.sigmoidCapTemperature = b.sigmoidCapTemperature;
        this.hcHead = b.hcHead;
        this.d2t = b.d2t;
        this.pooling = b.pooling;
        this.dense2 = b.dense2;
        this.dense3 = b.dense3;
        this.cls = b.cls;
        this.clsOutput = b.clsOutput;
        this.clsNorm = b.clsNorm;
        this.colbertOutput = b.colbertOutput;
        this.colbertOutputBias = b.colbertOutputBias;
        this.sparseOutput = b.sparseOutput;
        this.sparseOutputBias = b.sparseOutputBias;
        this.classifierScope = b.classifierScope;
        this.embeddingProjection = b.embeddingProjection;
        this.embeddingOutput = b.embeddingOutput;
        this.embeddingOutputBias = b.embeddingOutputBias;
        this.maskedLogitsStart = b.maskedLogitsStart;
        this.maskedLogitsEnd = b.maskedLogitsEnd;
        this.outputNormGroups = b.outputNormGroups;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** {@code output_norm.weight}; null when the model has no final norm. */
    public TensorRef outputNorm() { return outputNorm; }
    public TensorRef outputNormBias() { return outputNormBias; }
    /** Equal row slices the output norm normalizes on their own before its full-width weight; 1 for a row norm. */
    public int outputNormGroups() { return outputNormGroups; }
    /**
     * Projection {@code {n_embd, output width}} applied to the normed hidden rows before the LM head (OPT's
     * {@code project_out} when {@code word_embed_proj_dim != hidden_size}); null when the head reads the normed rows.
     */
    public TensorRef outputProjection() { return outputProjection; }
    /** LM head weight ({@code output.weight}, or {@code token_embd.weight} when tied); null for pooled-only models. */
    public TensorRef output() { return output; }
    public boolean tied() { return tied; }
    public TensorRef outputBias() { return outputBias; }
    /** Final logit softcap; 0 disables. */
    public float finalSoftcap() { return finalSoftcap; }
    /** True when the logits end in {@code scale * sigmoid((x + shift) / temperature)} (limite.cpp:243-248). */
    public boolean sigmoidCapped() { return sigmoidCapScale != 0f && sigmoidCapTemperature != 0f; }
    /** Scale of the sigmoid logit cap. */
    public float sigmoidCapScale() { return sigmoidCapScale; }
    /** Shift of the sigmoid logit cap. */
    public float sigmoidCapShift() { return sigmoidCapShift; }
    /** Temperature of the sigmoid logit cap. */
    public float sigmoidCapTemperature() { return sigmoidCapTemperature; }
    /** Logit scale; 1 leaves the logits unchanged. */
    public float logitScale() { return logitScale; }
    /** True when logits are divided by {@link #logitScale()} (granite), false when multiplied. */
    public boolean logitScaleDivides() { return logitScaleDivides; }
    /** Per-vocabulary logit multiplier applied after the projection (Dory's {@code lm_head.logit_scale}); null when absent. */
    public TensorRef logitScaleVector() { return logitScaleVector; }
    public boolean hcHead() { return hcHead; }
    /** Draft-to-target id table ({@code d2t}) of a reduced-vocabulary drafter head; null over the full vocabulary. */
    public TensorRef d2t() { return d2t; }
    public PoolingKind pooling() { return pooling; }
    public TensorRef dense2() { return dense2; }
    public TensorRef dense3() { return dense3; }
    /** Classifier hidden projection ({@code cls.weight}); null when absent. */
    public TensorRef cls() { return cls; }
    /** Classifier output ({@code cls.output.weight}); null when absent. */
    public TensorRef clsOutput() { return clsOutput; }
    public TensorRef clsNorm() { return clsNorm; }
    /**
     * Per-token embedding projection {@code {n_embd, n_embd_out}} of a multi-output model ({@code cls.colbert};
     * vLLM's BGE-M3 {@code colbert_linear}); null when the model has none.
     */
    public TensorRef colbertOutput() { return colbertOutput; }
    /** Bias {@code {n_embd_out}} added after {@link #colbertOutput()}; null when absent. */
    public TensorRef colbertOutputBias() { return colbertOutputBias; }
    /**
     * Per-token score projection {@code {n_embd, n_out}} of a multi-output model ({@code cls.sparse}; vLLM's
     * BGE-M3 {@code sparse_linear}); null when the model has none.
     */
    public TensorRef sparseOutput() { return sparseOutput; }
    /** Bias {@code {n_out}} added after {@link #sparseOutput()}; null when absent. */
    public TensorRef sparseOutputBias() { return sparseOutputBias; }
    /**
     * Where the classifier chain runs ({@code classifier.scope}); null keeps llama.cpp's legacy classifier, which
     * runs only for RANK pooling. {@link ClassifierScope#SEQUENCE} runs it over the pooled row of any pooled type,
     * {@link ClassifierScope#TOKEN} over every token of NONE pooling.
     */
    public ClassifierScope classifierScope() { return classifierScope; }
    /**
     * Projection {@code {n_embd, n_embd_out}} applied to the normed hidden rows of an embedding model
     * ({@code t_embd = proj · output_norm(x)}); null when the hidden rows are the normed rows.
     */
    public TensorRef embeddingProjection() { return embeddingProjection; }
    /**
     * Projection {@code {n_embd, n_embd_out}} that forms the embedding rows from the normed hidden rows
     * ({@code t_embd = W · output_norm(x) + b}) for pooling and per-token embeddings, leaving the LM head on the
     * normed rows; null when the embedding rows are the hidden rows.
     */
    public TensorRef embeddingOutput() { return embeddingOutput; }
    /** Bias {@code {n_embd_out}} added after {@link #embeddingOutput()}; null when absent. */
    public TensorRef embeddingOutputBias() { return embeddingOutputBias; }
    /** First logit set to {@code -Float.MAX_VALUE}; equals {@link #maskedLogitsEnd()} when nothing is masked. */
    public int maskedLogitsStart() { return maskedLogitsStart; }
    /** End (exclusive) of the masked logit range. */
    public int maskedLogitsEnd() { return maskedLogitsEnd; }
    /** True when a logit range is masked. */
    public boolean masksLogits() { return maskedLogitsEnd > maskedLogitsStart; }

    /** Width of a hidden row after the head: {@link #embeddingProjection()}'s output width, else {@code nEmbd}. */
    public int hiddenWidth(int nEmbd) {
        return embeddingProjection != null ? (int) embeddingProjection.ne(1) : nEmbd;
    }

    /**
     * Width of a row the LM head reads: {@link #outputProjection()}'s output width when present, else {@code nEmbd}.
     * The head reads the normed rows, so the embedding projection does not change it (qwen35's LM head keeps reading
     * the hidden rows, models/qwen35.cpp:47-48, 222-229).
     */
    public int headInputWidth(int nEmbd) {
        return outputProjection != null ? (int) outputProjection.ne(1) : nEmbd;
    }

    /**
     * Width of an embedding row before pooling: {@link #embeddingOutput()}'s output width, else
     * {@link #hiddenWidth(int)}.
     */
    public int embeddingWidth(int nEmbd) {
        return embeddingOutput != null ? (int) embeddingOutput.ne(1) : hiddenWidth(nEmbd);
    }

    /** Rows of the LM head, the width of a logits row; 0 without an LM head. */
    public long vocabRows() {
        return output != null ? output.ne(1) : 0;
    }

    @Override
    public String toString() {
        return "HeadSpec{norm=" + (outputNorm != null) + ", output=" + (output != null ? output.name() : "none") + ", tied="
                + tied + ", softcap=" + finalSoftcap + ", logitScale=" + (logitScaleDivides ? "/" : "*") + logitScale
                + ", hc=" + hcHead + ", pooling=" + pooling
                + ", cls=" + (cls != null || clsOutput != null)
                + (colbertOutput != null ? ", colbert=" + colbertOutput.name() : "")
                + (sparseOutput != null ? ", sparse=" + sparseOutput.name() : "")
                + (outputProjection != null ? ", outProj=" + outputProjection.name() : "")
                + ", embdProj=" + (embeddingProjection != null ? embeddingProjection.name() : "none")
                + (embeddingOutput != null ? ", embdOut=" + embeddingOutput.name()
                        + (embeddingOutputBias != null ? "+bias" : "") : "")
                + (masksLogits() ? ", masked=[" + maskedLogitsStart + ", " + maskedLogitsEnd + ")" : "")
                + (outputNormGroups > 1 ? ", normGroups=" + outputNormGroups : "") + '}';
    }

    public static final class Builder {
        private TensorRef outputNorm;
        private TensorRef outputNormBias;
        private TensorRef outputProjection;
        private TensorRef output;
        private boolean tied;
        private TensorRef outputBias;
        private float finalSoftcap;
        private float sigmoidCapScale;
        private float sigmoidCapShift;
        private float sigmoidCapTemperature;
        private float logitScale = 1.0f;
        private boolean logitScaleDivides;
        private TensorRef logitScaleVector;
        private boolean hcHead;
        private TensorRef d2t;
        private PoolingKind pooling = PoolingKind.NONE;
        private TensorRef dense2;
        private TensorRef dense3;
        private TensorRef cls;
        private TensorRef clsOutput;
        private TensorRef clsNorm;
        private TensorRef colbertOutput;
        private TensorRef colbertOutputBias;
        private TensorRef sparseOutput;
        private TensorRef sparseOutputBias;
        private ClassifierScope classifierScope;
        private TensorRef embeddingProjection;
        private TensorRef embeddingOutput;
        private TensorRef embeddingOutputBias;
        private int maskedLogitsStart;
        private int maskedLogitsEnd;
        private int outputNormGroups = 1;

        private Builder() {
        }

        public Builder outputNorm(TensorRef weight, TensorRef bias) {
            this.outputNorm = weight;
            this.outputNormBias = bias;
            return this;
        }
        /** Projection applied to the normed rows before the LM head; null reads the normed rows directly. */
        public Builder outputProjection(TensorRef v) { this.outputProjection = v; return this; }
        public Builder output(TensorRef weight, boolean tied) { this.output = weight; this.tied = tied; return this; }
        public Builder outputBias(TensorRef v) { this.outputBias = v; return this; }
        public Builder finalSoftcap(float v) { this.finalSoftcap = v; return this; }
        /** The {@code final_logit_sigmoid_capping} as {scale, shift, temperature}; all zero disables. */
        public Builder finalSigmoidCapping(float[] cap) {
            if (cap == null || cap.length != 3) {
                throw new IllegalArgumentException("finalSigmoidCapping needs scale, shift and temperature, got "
                        + (cap == null ? "null" : cap.length + " values"));
            }
            this.sigmoidCapScale = cap[0];
            this.sigmoidCapShift = cap[1];
            this.sigmoidCapTemperature = cap[2];
            return this;
        }
        public Builder logitScale(float scale, boolean divides) {
            this.logitScale = scale;
            this.logitScaleDivides = divides;
            return this;
        }
        /** Per-vocabulary logit multiplier applied after the projection; null for none. */
        public Builder logitScaleVector(TensorRef v) { this.logitScaleVector = v; return this; }
        public Builder hcHead(boolean v) { this.hcHead = v; return this; }
        public Builder d2t(TensorRef v) { this.d2t = v; return this; }
        public Builder pooling(PoolingKind v) { this.pooling = Objects.requireNonNull(v, "pooling"); return this; }
        public Builder dense(TensorRef dense2, TensorRef dense3) {
            this.dense2 = dense2;
            this.dense3 = dense3;
            return this;
        }
        public Builder classifier(TensorRef cls, TensorRef clsOutput, TensorRef clsNorm) {
            this.cls = cls;
            this.clsOutput = clsOutput;
            this.clsNorm = clsNorm;
            return this;
        }
        public Builder classifierScope(ClassifierScope v) { this.classifierScope = v; return this; }
        /** The per-token embedding and score projections of a multi-output model (either may be null). */
        public Builder tokenHeads(TensorRef colbert, TensorRef colbertBias, TensorRef sparse,
                                  TensorRef sparseBias) {
            this.colbertOutput = colbert;
            this.colbertOutputBias = colbertBias;
            this.sparseOutput = sparse;
            this.sparseOutputBias = sparseBias;
            return this;
        }
        public Builder embeddingProjection(TensorRef v) { this.embeddingProjection = v; return this; }
        /** The embedding-row projection and its bias (either may be null; a bias needs the weight). */
        public Builder embeddingOutput(TensorRef weight, TensorRef bias) {
            this.embeddingOutput = weight;
            this.embeddingOutputBias = bias;
            return this;
        }
        public Builder outputNormGroups(int v) { this.outputNormGroups = v; return this; }
        /** Logits {@code [start, end)} set to {@code -Float.MAX_VALUE}; {@code start == end} masks nothing. */
        public Builder maskedLogits(int start, int end) {
            this.maskedLogitsStart = start;
            this.maskedLogitsEnd = end;
            return this;
        }

        /** @throws IllegalArgumentException on an invalid scale or softcap, or an inconsistent LM head */
        public HeadSpec build() {
            if (!(finalSoftcap >= 0f) || Float.isInfinite(finalSoftcap)) {
                throw new IllegalArgumentException("final softcap must be >= 0, got " + finalSoftcap);
            }
            if (sigmoidCapScale != 0f && sigmoidCapTemperature != 0f
                    && (!(sigmoidCapScale > 0f) || !(sigmoidCapTemperature > 0f)
                        || Float.isInfinite(sigmoidCapScale) || Float.isInfinite(sigmoidCapTemperature))) {
                throw new IllegalArgumentException("final sigmoid capping needs positive scale and temperature, got "
                        + sigmoidCapScale + " / " + sigmoidCapTemperature);
            }
            if (!(logitScale > 0f) || Float.isInfinite(logitScale)) {
                throw new IllegalArgumentException("logit scale must be positive, got " + logitScale);
            }
            if (outputNormBias != null && outputNorm == null) {
                throw new IllegalArgumentException("output_norm.bias without output_norm.weight");
            }
            if (tied && output == null) {
                throw new IllegalArgumentException("tied LM head without an output tensor");
            }
            if (outputNormGroups < 1 || (outputNormGroups > 1 && (outputNorm == null
                    || (outputNorm.ne(0) % outputNormGroups) != 0))) {
                throw new IllegalArgumentException("a grouped output norm needs output_norm.weight with a width"
                        + " divisible by the " + outputNormGroups + " groups");
            }
            if (embeddingProjection != null) {
                if (output != null) {
                    throw new IllegalArgumentException("an embedding projection replaces the LM head, but the head has "
                            + output.name());
                }
                if (embeddingProjection.rank() != 2 || embeddingProjection.ne(1) <= 0
                        || embeddingProjection.ne(1) > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("embedding projection must be a 2-D matrix, got "
                            + embeddingProjection);
                }
            }
            checkOutputProjection();
            checkEmbeddingOutput();
            if (classifierScope != null) {
                if (cls == null && clsOutput == null) {
                    throw new IllegalArgumentException("classifier scope " + classifierScope.ggufName()
                            + " needs a classifier projection (cls or cls.output)");
                }
                if (classifierScope == ClassifierScope.TOKEN && pooling != PoolingKind.NONE) {
                    throw new IllegalArgumentException("classifier scope token needs NONE pooling, got " + pooling);
                }
                if (classifierScope == ClassifierScope.SEQUENCE && pooling == PoolingKind.NONE) {
                    throw new IllegalArgumentException("classifier scope sequence needs a pooled type, got NONE");
                }
            }
            if (maskedLogitsStart < 0 || maskedLogitsEnd < maskedLogitsStart
                    || (maskedLogitsEnd > maskedLogitsStart && (output == null || maskedLogitsEnd > output.ne(1)))) {
                throw new IllegalArgumentException("masked logit range [" + maskedLogitsStart + ", " + maskedLogitsEnd
                        + ") must lie within the " + (output != null ? output.ne(1) : 0) + " LM head rows");
            }
            return new HeadSpec(this);
        }

        /** The output projection needs an LM head of its output width and excludes the embedding projection. */
        private void checkOutputProjection() {
            if (outputProjection == null) {
                return;
            }
            if (output == null) {
                throw new IllegalArgumentException("an output projection " + outputProjection.name()
                        + " needs an LM head");
            }
            if (embeddingProjection != null) {
                throw new IllegalArgumentException("an output projection " + outputProjection.name()
                        + " cannot follow the embedding projection " + embeddingProjection.name());
            }
            if (outputProjection.rank() != 2 || outputProjection.ne(0) <= 0 || outputProjection.ne(1) <= 0) {
                throw new IllegalArgumentException("output projection must be a 2-D matrix, got " + outputProjection);
            }
            if (output.ne(0) != outputProjection.ne(1)) {
                throw new IllegalArgumentException("LM head " + output.name() + " reads " + output.ne(0)
                        + " values per row but the output projection " + outputProjection.name() + " writes "
                        + outputProjection.ne(1));
            }
        }

        private void checkEmbeddingOutput() {
            if (embeddingOutputBias != null && embeddingOutput == null) {
                throw new IllegalArgumentException("embedding output bias " + embeddingOutputBias.name()
                        + " without its weight");
            }
            if (embeddingOutput == null) {
                return;
            }
            if (embeddingProjection != null) {
                throw new IllegalArgumentException("an embedding output " + embeddingOutput.name()
                        + " cannot follow the embedding projection " + embeddingProjection.name());
            }
            if (embeddingOutput.rank() != 2 || embeddingOutput.ne(1) <= 0
                    || embeddingOutput.ne(1) > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("embedding output must be a 2-D matrix, got " + embeddingOutput);
            }
            if (embeddingOutputBias != null && (embeddingOutputBias.ne(0) != embeddingOutput.ne(1)
                    || embeddingOutputBias.ne(1) != 1 || embeddingOutputBias.ne(2) != 1
                    || embeddingOutputBias.ne(3) != 1)) {
                throw new IllegalArgumentException("embedding output bias " + embeddingOutputBias + " must hold the "
                        + embeddingOutput.ne(1) + " outputs of " + embeddingOutput.name());
            }
        }
    }
}
