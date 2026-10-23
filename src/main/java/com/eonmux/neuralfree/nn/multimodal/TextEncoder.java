package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Transformer text encoder: token embedding, sinusoidal positions, pre-norm {@link EncoderBlock}s and a final layer
 * norm. Every step is a {@link TensorOps} composition (or a module built on one), so the encoder runs on the device
 * when the train kernels are available on the input's backend.
 *
 * <p>The sinusoidal table {@code pe[pos, i] = sin|cos(pos / 10000^(2*(i/2)/d))} (sin for even {@code i}) is computed
 * once on the host at initialization and held as the non-persistent buffer {@code positionalEncoding}, which the
 * configuration determines and {@link #stateDict()} leaves out. Position {@code s} of a sequence adds row
 * {@code s % maxSeqLen}.</p>
 *
 * <p><b>Embedding dropout</b> (the {@code drop} of GPT-2's embeddings and the dropout of PyTorch's positional
 * encoding): in training mode the summed token and position embeddings pass through the registered {@link Dropout}
 * child {@value #EMBEDDING_DROPOUT} with rate {@code dropout} before the blocks. Its keep decisions are host draws, one
 * per element in row-major order, through {@code dropout}'s seeded generator; {@link #initialize} seeds it from its
 * {@code Random} after every weight when {@code dropout > 0}, so equal seeds give equal masks on every backend. In
 * evaluation mode, or with {@code dropout == 0}, the embeddings pass unchanged. A training-mode dropout records
 * {@code MaskScaleGrad}, so a dropped element passes no gradient.</p>
 */
public class TextEncoder extends Module {

    /** Registered name of the embedding-dropout child module. */
    public static final String EMBEDDING_DROPOUT = "embeddingDropout";

    private static final String POSITIONAL_ENCODING = "positionalEncoding";

    private final int vocabSize;
    private final int hiddenDim;
    private final int numLayers;
    private final int numHeads;
    private final int maxSeqLen;
    private final float dropout;
    private final String backboneType;
    private final Dropout embeddingDropout;

    private Embedding tokenEmbed;
    private final List<EncoderBlock> encoderLayers;
    private LayerNorm finalNorm;

    private boolean initialized = false;

    public TextEncoder(int vocabSize, int hiddenDim, int numLayers, int numHeads, int maxSeqLen, float dropout, String backboneType) {
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be positive");
        if (hiddenDim <= 0) throw new IllegalArgumentException("hiddenDim must be positive");
        if (numLayers < 0) throw new IllegalArgumentException("numLayers must be non-negative");
        if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be positive");
        if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be positive");
        if (!(dropout >= 0 && dropout < 1)) throw new IllegalArgumentException("dropout must be in [0, 1): " + dropout);
        ScaledDotProductAttention.requireDivisible(hiddenDim, numHeads);

        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        this.backboneType = backboneType != null ? backboneType : "transformer";
        this.embeddingDropout = registerModule(EMBEDDING_DROPOUT, new Dropout(dropout));

        this.encoderLayers = new ArrayList<>();
    }

    public TextEncoder(int vocabSize, int hiddenDim, int numLayers, int numHeads, int maxSeqLen) {
        this(vocabSize, hiddenDim, numLayers, numHeads, maxSeqLen, 0.1f, "transformer");
    }

    /**
     * Creates the parameters on {@code backend}, drawing every weight from {@code random} (a new one when null), then,
     * when {@code dropout > 0}, the seed of the embedding-dropout masks.
     */
    public void initialize(ComputeBackend backend, Random random) {
        if (initialized) return;

        Random r = random != null ? random : new Random();

        tokenEmbed = new Embedding(vocabSize, hiddenDim);
        tokenEmbed.initialize(backend, r);
        registerModule("tokenEmbed", tokenEmbed);

        registerBuffer(POSITIONAL_ENCODING, sinusoidalTable(backend, maxSeqLen, hiddenDim), false);

        for (int i = 0; i < numLayers; i++) {
            encoderLayers.add(registerModule("layer_" + i, new EncoderBlock(backend, hiddenDim, numHeads, r)));
        }

        finalNorm = new LayerNorm(new int[]{hiddenDim});
        finalNorm.initialize(backend);
        registerModule("finalNorm", finalNorm);

        if (dropout > 0) {
            embeddingDropout.setSeed(r.nextLong());
        }
        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        return encode(input);
    }

    /**
     * Encodes {@code [batch, seqLen]} INT32 or INT64 token ids into {@code [batch, seqLen, hiddenDim]}. In training
     * mode the embeddings pass through the embedding dropout before the blocks.
     *
     * @throws IllegalArgumentException when the ids are not rank 2 or not an integer dtype
     */
    public Tensor encode(Tensor inputIds) {
        if (!initialized) {
            throw new IllegalStateException("TextEncoder not initialized. Call initialize() first.");
        }
        if (inputIds == null) {
            throw new IllegalArgumentException("inputIds tensor cannot be null");
        }
        if (inputIds.shape().rank() != 2) {
            throw new IllegalArgumentException("inputIds must be [batch, seqLen], got " + inputIds.shape());
        }

        Tensor summed;
        try (Tensor embedded = tokenEmbed.forward(inputIds);
             Tensor positions = positionRows(inputIds.shape().dim(1))) {
            summed = TensorOps.add(embedded, positions);
        }
        Tensor hidden = dropEmbeddings(summed);
        for (EncoderBlock layer : encoderLayers) {
            try (Tensor previous = hidden) {
                hidden = layer.forward(previous);
            }
        }
        try (Tensor last = hidden) {
            return finalNorm.forward(last);
        }
    }

    /**
     * Pools {@code [batch, seqLen, hiddenDim]} encoder output: {@code "cls"}/{@code "first"} take position 0,
     * {@code "mean"} and {@code "max"} reduce the sequence axis; any other method returns {@code encoderOutput}
     * itself. Every other result is a new tensor owned by the caller.
     */
    public Tensor poolOutput(Tensor encoderOutput, String poolMethod) {
        if ("cls".equals(poolMethod) || "first".equals(poolMethod)) {
            int[] dims = encoderOutput.shape().dims();
            try (Tensor first = encoderOutput.slice(1, 0, 1)) {
                return first.reshape(Shape.of(dims[0], dims[2]));
            }
        } else if ("mean".equals(poolMethod)) {
            return TensorOps.mean(encoderOutput, 1);
        } else if ("max".equals(poolMethod)) {
            return TensorOps.max(encoderOutput, 1);
        }
        return encoderOutput;
    }

    /**
     * {@code embeddings} through the embedding dropout; the result is owned by the caller and {@code embeddings} is
     * closed when the dropout made a new tensor (or failed).
     */
    private Tensor dropEmbeddings(Tensor embeddings) {
        Tensor dropped;
        try {
            dropped = embeddingDropout.forward(embeddings);
        } catch (RuntimeException e) {
            embeddings.close();
            throw e;
        }
        if (dropped != embeddings) {
            embeddings.close();
        }
        return dropped;
    }

    /** The positional rows {@code s % maxSeqLen} for {@code s < seqLen}, as a {@code [seqLen, hiddenDim]} tensor. */
    private Tensor positionRows(int seqLen) {
        Tensor table = buffer(POSITIONAL_ENCODING);
        if (seqLen <= maxSeqLen) {
            return table.slice(0, 0, seqLen);
        }
        int[] rows = new int[seqLen];
        for (int s = 0; s < seqLen; s++) {
            rows[s] = s % maxSeqLen;
        }
        try (Tensor index = Tensor.fromArray(table.backend(), rows)) {
            return TensorOps.indexSelectRows(table, index, false);
        }
    }

    private static Tensor sinusoidalTable(ComputeBackend backend, int maxSeqLen, int dModel) {
        float[] peData = new float[Math.multiplyExact(maxSeqLen, dModel)];
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dModel; i++) {
                float angle = pos / (float) Math.pow(10000, (2 * (i / 2)) / (float) dModel);
                peData[pos * dModel + i] = (float) (i % 2 == 0 ? Math.sin(angle) : Math.cos(angle));
            }
        }
        try (Tensor flat = Tensor.fromArray(backend, peData)) {
            return flat.reshape(Shape.of(maxSeqLen, dModel));
        }
    }

    public int vocabSize() { return vocabSize; }
    public int hiddenDim() { return hiddenDim; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int maxSeqLen() { return maxSeqLen; }
    public float dropout() { return dropout; }
    public String backboneType() { return backboneType; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("TextEncoder(vocabSize=%d, hiddenDim=%d, numLayers=%d, numHeads=%d, maxSeqLen=%d, dropout=%.2f, backboneType=%s)",
                vocabSize, hiddenDim, numLayers, numHeads, maxSeqLen, dropout, backboneType);
    }
}
