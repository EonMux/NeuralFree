package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.activations.GELU;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Masked language modelling with BERT's output head. Masks are drawn on the host from the module's {@link Random}
 * and enter the device as row indices; the masked hidden states, the head, the loss and the
 * predictions are module and {@link TensorOps} compositions on the encoder output's backend, with nothing read back.
 *
 * <p><b>Output head</b> (BERT's {@code BertLMPredictionHead}), registered as the children {@code dense},
 * {@code activation}, {@code layer_norm} and {@code decoder}: for the encoder output {@code h} of a masked position,
 * {@code logits = decoder(LayerNorm(GELU(dense(h))))} with {@code dense} a {@link Linear}{@code (H, H)}, the
 * {@link GELU} module (the tanh approximation of {@link TensorOps#gelu}), a {@link LayerNorm}{@code (H)} with
 * {@code eps = 1e-12} and an affine step, and {@code decoder} computing {@code x W^T + b} with {@code W}
 * {@code [vocabSize, H]} and a bias {@code b} {@code [vocabSize]}. Untied (the default), the decoder is a
 * {@link Linear}{@code (H, vocabSize)}. Tied, {@code W} is a {@code [vocabSize, H]} parameter of the encoder (its
 * token embedding table) passed at construction: the decoder reads it without registering it, so it is saved, loaded,
 * moved and closed with the encoder, appears once in {@link #parameters()}, and accumulates the gradient of both
 * uses; the decoder registers only its bias.</p>
 *
 * <p><b>Initialization.</b> {@link #initialize(ComputeBackend)} creates the head's parameters from the module's
 * {@link Random} ({@link #setRandom(Random)}): {@code dense} and an untied {@code decoder} by
 * {@link Linear#initialize(ComputeBackend, Random)} (zero biases), the layer norm with weight 1 and bias 0, a tied
 * decoder's bias as zeros.</p>
 *
 * <p><b>Autograd.</b> The logits and the loss are recorded when grad mode is on and a parameter of the encoder or of
 * the head requires grad; the gradient reaches the head and, through the selected rows of the encoder output, the
 * encoder. The predictions (an argmax) are computed under {@link ComputationGraph#noGrad()} and never recorded.</p>
 */
public final class MaskedLanguageModel extends Module {

    /** The layer-norm epsilon of BERT's prediction head ({@code layer_norm_eps}). */
    public static final float LAYER_NORM_EPSILON = 1e-12f;

    /**
     * The result of {@link #forwardMLM}, owned by the caller. {@link #mlmLoss()} is the recorded loss to
     * back-propagate. {@link #logits()} is an output of the same recorded graph, valid until the loss is
     * back-propagated (whose backward pass closes it); take a {@link Tensor#detach()} handle first to read it
     * afterwards. {@link #predictions()} (an argmax, never recorded) stays valid.
     */
    public static final class MLMOutput {
        private final Tensor logits;
        private final Tensor predictions;
        private final int[] maskedPositions;
        private final Tensor mlmLoss;

        public MLMOutput(Tensor logits, Tensor predictions, int[] maskedPositions, Tensor mlmLoss) {
            this.logits = Objects.requireNonNull(logits);
            this.predictions = Objects.requireNonNull(predictions);
            this.maskedPositions = maskedPositions;
            this.mlmLoss = mlmLoss;
        }

        public Tensor logits() { return logits; }
        public Tensor predictions() { return predictions; }
        public int[] maskedPositions() { return maskedPositions; }
        public Tensor mlmLoss() { return mlmLoss; }
    }

    public static final class MLMMask {
        private final int[] maskedIndices;
        private final int[] originalTokens;
        private final int sequenceLength;
        private final float maskRatio;

        public MLMMask(int[] maskedIndices, int[] originalTokens, int sequenceLength, float maskRatio) {
            this.maskedIndices = maskedIndices;
            this.originalTokens = originalTokens;
            this.sequenceLength = sequenceLength;
            this.maskRatio = maskRatio;
        }

        public int[] maskedIndices() { return maskedIndices; }
        public int[] originalTokens() { return originalTokens; }
        public int sequenceLength() { return sequenceLength; }
        public float maskRatio() { return maskRatio; }
        public int numMasked() { return maskedIndices.length; }
    }

    private final Module encoder;
    private final int vocabSize;
    private final int hiddenSize;
    private final float maskProbability;
    private final int maskTokenId;
    private final Linear dense;
    private final GELU activation;
    private final LayerNorm layerNorm;
    /** The untied decoder; null when the decoder is tied. */
    private final Linear untiedDecoder;
    /** The tied decoder; null when the decoder is untied. */
    private final TiedDecoder tiedDecoder;
    private Random random;
    private boolean initialized = false;
    private volatile boolean closed = false;

    public MaskedLanguageModel(Module encoder, int vocabSize, int hiddenSize) {
        this(encoder, vocabSize, hiddenSize, 0.15f, 103, null);
    }

    /** A model whose decoder weight is tied to {@code tieTo} (see the class description). */
    public MaskedLanguageModel(Module encoder, int vocabSize, int hiddenSize, Parameter tieTo) {
        this(encoder, vocabSize, hiddenSize, 0.15f, 103, Objects.requireNonNull(tieTo, "tieTo cannot be null"));
    }

    public MaskedLanguageModel(Module encoder, int vocabSize, int hiddenSize,
                                float maskProbability, int maskTokenId) {
        this(encoder, vocabSize, hiddenSize, maskProbability, maskTokenId, null);
    }

    /**
     * @param tieTo the encoder parameter the decoder weight is tied to, {@code [vocabSize, hiddenSize]}; null for an
     *              untied decoder
     * @throws NullPointerException     when {@code encoder} is null
     * @throws IllegalArgumentException when a size is not positive, {@code maskProbability} is outside
     *                                  {@code (0, 1)}, or {@code tieTo} is not a {@code [vocabSize, hiddenSize]}
     *                                  parameter of the encoder
     */
    public MaskedLanguageModel(Module encoder, int vocabSize, int hiddenSize,
                                float maskProbability, int maskTokenId, Parameter tieTo) {
        Objects.requireNonNull(encoder, "Encoder cannot be null");
        if (vocabSize <= 0) throw new IllegalArgumentException("Vocab size must be positive");
        if (hiddenSize <= 0) throw new IllegalArgumentException("Hidden size must be positive");
        if (maskProbability <= 0 || maskProbability >= 1) {
            throw new IllegalArgumentException("Mask probability must be in (0, 1)");
        }
        if (tieTo != null) {
            requireTiedWeight(encoder, tieTo, vocabSize, hiddenSize);
        }

        this.encoder = registerModule("encoder", encoder);
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.maskProbability = maskProbability;
        this.maskTokenId = maskTokenId;
        this.random = new Random();
        this.dense = registerModule("dense", new Linear(hiddenSize, hiddenSize));
        this.activation = registerModule("activation", new GELU());
        this.layerNorm = registerModule("layer_norm", new LayerNorm(hiddenSize, LAYER_NORM_EPSILON, true));
        if (tieTo == null) {
            this.untiedDecoder = registerModule("decoder", new Linear(hiddenSize, vocabSize));
            this.tiedDecoder = null;
        } else {
            this.untiedDecoder = null;
            this.tiedDecoder = registerModule("decoder", new TiedDecoder(tieTo, vocabSize));
        }
    }

    private static void requireTiedWeight(Module encoder, Parameter tieTo, int vocabSize, int hiddenSize) {
        Shape expected = Shape.of(vocabSize, hiddenSize);
        if (!tieTo.shape().equals(expected)) {
            throw new IllegalArgumentException("The tied decoder weight must have shape " + expected + ", got "
                + tieTo.shape());
        }
        for (Parameter p : encoder.parameters()) {
            if (p == tieTo) {
                return;
            }
        }
        throw new IllegalArgumentException("The tied decoder weight must be a parameter of the encoder (its token "
            + "embedding table), so the encoder owns, saves and moves it");
    }

    /**
     * Creates the head's parameters on {@code backend} from the module's {@link Random} (see the class description).
     * Records nothing.
     *
     * @throws IllegalStateException    when the head is already initialized or the module is closed
     * @throws IllegalArgumentException when the tied decoder weight lives on another backend
     */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        ensureOpen();
        if (initialized) {
            throw new IllegalStateException("MaskedLanguageModel head already initialized");
        }
        if (tiedDecoder != null && tiedDecoder.weight.backend() != backend) {
            throw new IllegalArgumentException("The tied decoder weight lives on " + tiedDecoder.weight.backend()
                + ", not on " + backend);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            dense.initialize(backend, random);
            layerNorm.initialize(backend);
            if (untiedDecoder != null) {
                untiedDecoder.initialize(backend, random);
            } else {
                tiedDecoder.initialize(backend);
            }
        }
        initialized = true;
    }

    public MLMMask createMask(int[] tokens, int sequenceLength) {
        Objects.requireNonNull(tokens, "Tokens cannot be null");

        int batchSize = tokens.length / sequenceLength;
        List<Integer> allMaskedIndices = new ArrayList<>();
        List<Integer> allOriginalTokens = new ArrayList<>();

        for (int b = 0; b < batchSize; b++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 1; i < sequenceLength - 1; i++) {
                candidates.add(b * sequenceLength + i);
            }
            Collections.shuffle(candidates, random);

            int numToMask = Math.max(1, (int) ((sequenceLength - 2) * maskProbability));
            for (int i = 0; i < Math.min(numToMask, candidates.size()); i++) {
                int idx = candidates.get(i);
                allMaskedIndices.add(idx);
                allOriginalTokens.add(tokens[idx]);
            }
        }

        int[] maskedIndices = allMaskedIndices.stream().mapToInt(Integer::intValue).toArray();
        int[] originalTokens = allOriginalTokens.stream().mapToInt(Integer::intValue).toArray();

        return new MLMMask(maskedIndices, originalTokens, sequenceLength, maskProbability);
    }

    public int[] applyMask(int[] tokens, MLMMask mask) {
        Objects.requireNonNull(tokens, "Tokens cannot be null");
        Objects.requireNonNull(mask, "Mask cannot be null");

        int[] maskedTokens = tokens.clone();
        for (int idx : mask.maskedIndices()) {
            float r = random.nextFloat();
            if (r < 0.8f) {
                maskedTokens[idx] = maskTokenId;
            } else if (r < 0.9f) {
                maskedTokens[idx] = random.nextInt(vocabSize);
            }
        }
        return maskedTokens;
    }

    /**
     * Encodes {@code inputIds} {@code [batch, seqLen]} and scores the masked positions. {@code logits} is
     * {@code [numMasked, vocabSize]}, the output head of the class description applied to the encoder output rows of
     * the masked positions; {@code predictions} holds the FLOAT32 index of each row's first maximum; {@code mlmLoss}
     * is the cross-entropy {@code -mean_i(logSoftmax(logits)[i, originalTokens[i]])} over the masked positions.
     *
     * @throws IllegalStateException     when the head is not initialized or the module is closed
     * @throws IllegalArgumentException  when the mask is empty, the mask arrays differ in length, or the encoder
     *                                   output does not hold {@code [batch, seqLen, hiddenSize]} values
     * @throws IndexOutOfBoundsException when a masked position is outside {@code [0, batch * seqLen)} or an original
     *                                   token outside {@code [0, vocabSize)}
     */
    public MLMOutput forwardMLM(Tensor inputIds, MLMMask mask) {
        ensureOpen();
        ensureInitialized();
        Objects.requireNonNull(inputIds, "Input IDs cannot be null");
        Objects.requireNonNull(mask, "Mask cannot be null");

        int batchSize = inputIds.shape().dim(0);
        int seqLen = inputIds.shape().dim(1);
        int numMasked = mask.numMasked();
        int[] positions = mask.maskedIndices();
        int[] targets = mask.originalTokens();
        validateMask(positions, targets, batchSize * seqLen);

        Tensor encoded = null;
        Tensor logits = null;
        Tensor predictions = null;
        try {
            encoded = encoder.forward(inputIds);
            ComputeBackend backend = encoded.backend();
            try (Tensor positionIndex = Tensor.fromArray(backend, positions);
                 Tensor hidden = hiddenRows(encoded, batchSize, seqLen);
                 Tensor maskedHidden = TensorOps.indexSelectRows(hidden, positionIndex, false)) {
                logits = head(maskedHidden);
            }
            predictions = predict(logits);
            try (Tensor targetVector = Tensor.fromArray(backend, targets);
                 Tensor targetIndex = targetVector.reshape(numMasked, 1);
                 Tensor logProbs = TensorOps.logSoftmax(logits, 1);
                 Tensor targetLogProbs = TensorOps.gather(logProbs, 1, targetIndex);
                 Tensor meanLogProb = TensorOps.mean(targetLogProbs)) {
                Tensor loss = TensorOps.neg(meanLogProb);
                return new MLMOutput(logits, predictions, positions.clone(), loss);
            }
        } catch (RuntimeException e) {
            release(logits);
            release(predictions);
            throw e;
        } finally {
            release(encoded, inputIds);
        }
    }

    /** {@code decoder(LayerNorm(GELU(dense(rows))))} for {@code [n, hiddenSize]} rows, as a new tensor. */
    private Tensor head(Tensor rows) {
        try (Tensor projected = dense.forward(rows);
             Tensor activated = activation.forward(projected);
             Tensor normalized = layerNorm.forward(activated)) {
            return untiedDecoder != null ? untiedDecoder.forward(normalized) : tiedDecoder.forward(normalized);
        }
    }

    /** The FLOAT32 index of each row's first maximum of {@code logits}, computed under no-grad. */
    private static Tensor predict(Tensor logits) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor best = TensorOps.argmax(logits, 1)) {
            return best.to(DType.FLOAT32);
        }
    }

    /** A {@code [batch * seqLen, hiddenSize]} view of the encoder output. */
    private Tensor hiddenRows(Tensor encoded, int batchSize, int seqLen) {
        if (encoded.numElements() != (long) batchSize * seqLen * hiddenSize) {
            throw new IllegalArgumentException("Encoder output of shape " + encoded.shape() + " does not hold ["
                + batchSize + ", " + seqLen + ", " + hiddenSize + "] values");
        }
        return encoded.reshape(batchSize * seqLen, hiddenSize);
    }

    private void validateMask(int[] positions, int[] targets, int positionCount) {
        if (positions.length == 0) {
            throw new IllegalArgumentException("The mask selects no positions");
        }
        if (targets.length != positions.length) {
            throw new IllegalArgumentException("The mask has " + positions.length + " positions but "
                + targets.length + " original tokens");
        }
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] < 0 || positions[i] >= positionCount) {
                throw new IndexOutOfBoundsException("Masked position " + positions[i] + " out of bounds for "
                    + positionCount + " positions");
            }
            if (targets[i] < 0 || targets[i] >= vocabSize) {
                throw new IndexOutOfBoundsException("Original token " + targets[i] + " out of bounds for vocabulary "
                    + "size " + vocabSize);
            }
        }
    }

    /** Closes {@code t} unless it is one of {@code keep}. */
    private static void release(Tensor t, Tensor... keep) {
        if (t == null) {
            return;
        }
        for (Tensor k : keep) {
            if (t == k) {
                return;
            }
        }
        t.close();
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        return encoder.forward(input);
    }

    public Module encoder() { return encoder; }
    public int vocabSize() { return vocabSize; }
    public int hiddenSize() { return hiddenSize; }
    public float maskProbability() { return maskProbability; }
    public int maskTokenId() { return maskTokenId; }
    /** The head's {@code Linear(H, H)} transform. */
    public Linear dense() { return dense; }
    /** The head's layer norm ({@code eps = 1e-12}). */
    public LayerNorm layerNorm() { return layerNorm; }
    /** The head's decoder module: a {@link Linear} when untied. */
    public Module decoder() { return untiedDecoder != null ? untiedDecoder : tiedDecoder; }
    /** Whether the decoder weight is tied to an encoder parameter. */
    public boolean isDecoderTied() { return tiedDecoder != null; }

    /** The decoder weight {@code [vocabSize, hiddenSize]}: the tied encoder parameter when tied. */
    public Parameter decoderWeight() {
        ensureInitialized();
        return untiedDecoder != null ? untiedDecoder.weight() : tiedDecoder.weight;
    }

    /** The decoder bias {@code [vocabSize]}. */
    public Parameter decoderBias() {
        ensureInitialized();
        return untiedDecoder != null ? untiedDecoder.bias() : tiedDecoder.bias();
    }

    /** Replaces the host random stream of the masks and of {@link #initialize(ComputeBackend)}. */
    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random, "Random cannot be null");
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("MaskedLanguageModel has been closed");
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException(
                "MaskedLanguageModel head not initialized. Call initialize(backend) first.");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) return "MaskedLanguageModel{closed=true}";
        return "MaskedLanguageModel{vocabSize=" + vocabSize +
               ", hiddenSize=" + hiddenSize +
               ", maskProbability=" + maskProbability +
               ", tiedDecoder=" + isDecoderTied() + "}";
    }

    /**
     * The decoder of a tied head: {@code x W^T + b} with {@code W} a borrowed encoder parameter (not registered here;
     * the encoder owns it) and its own registered bias {@code b}, zeros on initialization.
     */
    private static final class TiedDecoder extends Module {
        private final Parameter weight;
        private final int vocabSize;
        private Parameter bias;

        TiedDecoder(Parameter weight, int vocabSize) {
            this.weight = weight;
            this.vocabSize = vocabSize;
        }

        void initialize(ComputeBackend backend) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                bias = registerParameter("bias", new Parameter(Tensor.zeros(backend, Shape.vector(vocabSize)),
                    "bias"));
            }
        }

        Parameter bias() {
            return bias;
        }

        /**
         * {@code input @ W^T + b}; the transposed weight is a recorded view, so in grad mode the gradient reaches the
         * tied parameter and the bias.
         */
        @Override
        public Tensor forward(Tensor input) {
            Objects.requireNonNull(input, "Input cannot be null");
            if (bias == null) {
                throw new IllegalStateException("Tied decoder not initialized");
            }
            Tensor product;
            try (Tensor weightT = weight.tensor().transpose()) {
                product = TensorOps.matmul(input, weightT);
            }
            try {
                return TensorOps.add(product, bias.tensor());
            } finally {
                product.close();
            }
        }

        @Override
        public String toString() {
            return "TiedDecoder{vocabSize=" + vocabSize + ", initialized=" + (bias != null) + '}';
        }
    }
}
