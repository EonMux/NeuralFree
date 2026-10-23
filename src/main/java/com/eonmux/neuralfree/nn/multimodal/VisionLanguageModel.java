package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.Random;

/**
 * Vision-language model: a {@link VisionEncoder} and a {@link TextEncoder}, projections to a shared width,
 * text-to-image {@link CrossAttentionFusion} layers, a final norm and a language-model head. The forward pass runs on
 * the device when the train kernels are available on the inputs' backend; intermediates are released before
 * returning.
 *
 * <p><b>Training.</b> {@link #forward(Tensor, Tensor)}, {@link #encodeVision} and {@link #encodeText} are recorded
 * compositions of the child modules, so a loss on their results reaches every parameter they read.
 * {@link #generate} is inference: it runs under {@link ComputationGraph#noGrad()} and records nothing.</p>
 */
public class VisionLanguageModel extends Module {

    private final int vocabSize;
    private final int imageHiddenDim;
    private final int textHiddenDim;
    private final int fusionHiddenDim;
    private final int numFusionLayers;
    private final int numHeads;
    private final int maxSeqLen;

    private VisionEncoder visionEncoder;
    private TextEncoder textEncoder;
    private Linear visionProjection;
    private Linear textProjection;
    private final CrossAttentionFusion[] fusionLayers;
    private LayerNorm finalNorm;
    private Linear lmHead;

    private boolean initialized = false;

    public VisionLanguageModel(int vocabSize, int imageHiddenDim, int textHiddenDim,
                               int fusionHiddenDim, int numFusionLayers, int numHeads, int maxSeqLen) {
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be positive");
        if (imageHiddenDim <= 0) throw new IllegalArgumentException("imageHiddenDim must be positive");
        if (textHiddenDim <= 0) throw new IllegalArgumentException("textHiddenDim must be positive");
        if (fusionHiddenDim <= 0) throw new IllegalArgumentException("fusionHiddenDim must be positive");
        if (numFusionLayers < 0) throw new IllegalArgumentException("numFusionLayers must be non-negative");
        if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be positive");
        if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be positive");
        ScaledDotProductAttention.requireDivisible(imageHiddenDim, numHeads);
        ScaledDotProductAttention.requireDivisible(textHiddenDim, numHeads);
        ScaledDotProductAttention.requireDivisible(fusionHiddenDim, numHeads);

        this.vocabSize = vocabSize;
        this.imageHiddenDim = imageHiddenDim;
        this.textHiddenDim = textHiddenDim;
        this.fusionHiddenDim = fusionHiddenDim;
        this.numFusionLayers = numFusionLayers;
        this.numHeads = numHeads;
        this.maxSeqLen = maxSeqLen;

        this.fusionLayers = new CrossAttentionFusion[numFusionLayers];
    }

    /**
     * Creates the encoders, projections, fusion layers and head on {@code backend}, drawing every weight from
     * {@code random} (a new one when null).
     */
    public void initialize(ComputeBackend backend, Random random,
                           int imageSize, int patchSize, int inChannels, int visionNumLayers,
                           int textNumLayers) {
        if (initialized) return;

        Random r = random != null ? random : new Random();

        visionEncoder = new VisionEncoder(imageSize, patchSize, inChannels, imageHiddenDim, visionNumLayers, numHeads);
        visionEncoder.initialize(backend, r);
        registerModule("visionEncoder", visionEncoder);

        textEncoder = new TextEncoder(vocabSize, textHiddenDim, textNumLayers, numHeads, maxSeqLen);
        textEncoder.initialize(backend, r);
        registerModule("textEncoder", textEncoder);

        visionProjection = new Linear(imageHiddenDim, fusionHiddenDim);
        visionProjection.initialize(backend, r);
        registerModule("visionProjection", visionProjection);

        textProjection = new Linear(textHiddenDim, fusionHiddenDim);
        textProjection.initialize(backend, r);
        registerModule("textProjection", textProjection);

        for (int i = 0; i < numFusionLayers; i++) {
            fusionLayers[i] = new CrossAttentionFusion(fusionHiddenDim, fusionHiddenDim, fusionHiddenDim, fusionHiddenDim, numHeads);
            fusionLayers[i].initialize(backend, r);
            registerModule("fusionLayer_" + i, fusionLayers[i]);
        }

        finalNorm = new LayerNorm(new int[]{fusionHiddenDim});
        finalNorm.initialize(backend);
        registerModule("finalNorm", finalNorm);

        lmHead = new Linear(fusionHiddenDim, vocabSize);
        lmHead.initialize(backend, r);
        registerModule("lmHead", lmHead);

        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("VisionLanguageModel requires both image and text inputs. Use forward(image, inputIds) method.");
    }

    /**
     * Logits {@code [batch, seqLen, vocabSize]} for a {@code [batch, channels, height, width]} image and
     * {@code [batch, seqLen]} INT32 or INT64 token ids: the projected text features attend to the projected vision
     * features through every fusion layer, then the final norm and the language-model head.
     */
    public Tensor forward(Tensor image, Tensor inputIds) {
        if (!initialized) {
            throw new IllegalStateException("VisionLanguageModel not initialized. Call initialize() first.");
        }

        Tensor projectedVision;
        try (Tensor visionFeatures = visionEncoder.encode(image)) {
            projectedVision = visionProjection.forward(visionFeatures);
        }
        try (Tensor vision = projectedVision) {
            Tensor fused;
            try (Tensor textFeatures = textEncoder.encode(inputIds)) {
                fused = textProjection.forward(textFeatures);
            }
            for (CrossAttentionFusion fusionLayer : fusionLayers) {
                try (Tensor previous = fused) {
                    fused = fusionLayer.forward(previous, vision, vision, null);
                }
            }
            try (Tensor last = fused;
                 Tensor normalized = finalNorm.forward(last)) {
                return lmHead.forward(normalized);
            }
        }
    }

    /**
     * Extends {@code inputIds} ({@code [batch, seqLen]}, INT32 or INT64) by {@code maxLength} sampled tokens and
     * returns the {@code [batch, seqLen + maxLength]} INT32 ids as a new tensor owned by the caller; {@code inputIds}
     * is never closed. Each step runs the model, takes the last position's {@code softmax(logits / temperature)} on
     * the device and samples one token per row from it with {@link Math#random()}; the {@code [batch, vocabSize]}
     * probabilities are the one readback per step, since the sampled tokens drive the Java loop.
     * Runs under {@link ComputationGraph#noGrad()}: nothing is recorded and the result does not require grad.
     *
     * @throws IllegalArgumentException when {@code temperature} is not positive, {@code maxLength} is negative, or
     *                                  the ids are not a rank-2 INT32 or INT64 tensor
     */
    public Tensor generate(Tensor image, Tensor inputIds, int maxLength, float temperature) {
        if (!initialized) {
            throw new IllegalStateException("VisionLanguageModel not initialized. Call initialize() first.");
        }
        if (!(temperature > 0.0f)) {
            throw new IllegalArgumentException("temperature must be positive, got " + temperature);
        }
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be non-negative, got " + maxLength);
        }
        if (inputIds == null || inputIds.shape().rank() != 2) {
            throw new IllegalArgumentException("inputIds must be [batch, seqLen]");
        }
        if (inputIds.dtype() != DType.INT32 && inputIds.dtype() != DType.INT64) {
            throw new IllegalArgumentException("inputIds must be INT32 or INT64, got " + inputIds.dtype());
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return sampleTokens(image, inputIds, maxLength, temperature);
        }
    }

    /** {@link #generate}'s loop, run by it under {@link ComputationGraph#noGrad()}. */
    private Tensor sampleTokens(Tensor image, Tensor inputIds, int maxLength, float temperature) {
        if (maxLength == 0) {
            Tensor ids32 = inputIds.to(DType.INT32);
            return ids32 != inputIds ? ids32 : inputIds.reshape(inputIds.shape());
        }

        Tensor currentIds = inputIds;
        try {
            for (int i = 0; i < maxLength; i++) {
                int batchSize = currentIds.shape().dim(0);
                float[] probs;
                try (Tensor logits = forward(image, currentIds)) {
                    probs = lastPositionProbabilities(logits, temperature);
                }
                int[] nextTokens = new int[batchSize];
                for (int b = 0; b < batchSize; b++) {
                    nextTokens[b] = sample(probs, b * vocabSize);
                }
                Tensor extended = appendTokens(currentIds, nextTokens);
                if (currentIds != inputIds) {
                    currentIds.close();
                }
                currentIds = extended;
            }
            return currentIds;
        } catch (RuntimeException e) {
            if (currentIds != inputIds) {
                currentIds.close();
            }
            throw e;
        }
    }

    /** {@code softmax(logits[:, last, :] / temperature)} read back as {@code [batch, vocabSize]}. */
    private static float[] lastPositionProbabilities(Tensor logits, float temperature) {
        int seqLen = logits.shape().dim(1);
        try (Tensor last = logits.slice(1, seqLen - 1, seqLen);
             Tensor t = Tensor.scalar(logits.backend(), temperature);
             Tensor scaled = TensorOps.div(last, t);
             Tensor probs = TensorOps.softmax(scaled, 2)) {
            return probs.toFloatArray();
        }
    }

    /**
     * {@code ids} ({@code [batch, len]}, INT32 or INT64) with one more column holding {@code tokens}, as a new INT32
     * tensor on the ids' backend. With the train kernels the ids are copied on the device into the fresh result
     * (called only from {@link #sampleTokens}, under {@link #generate}'s {@code noGrad} scope); otherwise (the CPU
     * path) they are read, extended and uploaded.
     */
    private static Tensor appendTokens(Tensor ids, int[] tokens) {
        ComputeBackend backend = ids.backend();
        int batchSize = ids.shape().dim(0);
        int len = ids.shape().dim(1);
        Shape extendedShape = Shape.of(batchSize, len + 1);
        Tensor ids32 = ids.to(DType.INT32);
        try {
            if (TrainGpu.forBackend(backend) != null) {
                Tensor extended = Tensor.zeros(backend, extendedShape, DType.INT32);
                try (Tensor head = extended.slice(1, 0, len);
                     Tensor tail = extended.slice(1, len, len + 1);
                     Tensor column = intTensor(backend, Shape.of(batchSize, 1), tokens)) {
                    GpuCoreOps.copyInto(ids32, head);
                    GpuCoreOps.copyInto(column, tail);
                    return extended;
                } catch (RuntimeException e) {
                    extended.close();
                    throw e;
                }
            }
            int[] previous = ids32.toIntArray();
            int[] next = new int[batchSize * (len + 1)];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(previous, b * len, next, b * (len + 1), len);
                next[b * (len + 1) + len] = tokens[b];
            }
            return intTensor(backend, extendedShape, next);
        } finally {
            if (ids32 != ids) {
                ids32.close();
            }
        }
    }

    private static Tensor intTensor(ComputeBackend backend, Shape shape, int[] values) {
        try (Tensor flat = Tensor.fromArray(backend, values)) {
            return flat.reshape(shape);
        }
    }

    /** Inverse-CDF sample from the {@code vocabSize} probabilities at {@code offset}. */
    private int sample(float[] probs, int offset) {
        double r = Math.random();
        double cumProb = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            cumProb += probs[offset + i];
            if (r < cumProb) return i;
        }
        return vocabSize - 1;
    }

    public Tensor encodeVision(Tensor image) {
        if (!initialized) {
            throw new IllegalStateException("VisionLanguageModel not initialized. Call initialize() first.");
        }
        try (Tensor features = visionEncoder.encode(image)) {
            return visionProjection.forward(features);
        }
    }

    public Tensor encodeText(Tensor inputIds) {
        if (!initialized) {
            throw new IllegalStateException("VisionLanguageModel not initialized. Call initialize() first.");
        }
        try (Tensor features = textEncoder.encode(inputIds)) {
            return textProjection.forward(features);
        }
    }

    public VisionEncoder visionEncoder() { return visionEncoder; }
    public TextEncoder textEncoder() { return textEncoder; }
    public int vocabSize() { return vocabSize; }
    public int imageHiddenDim() { return imageHiddenDim; }
    public int textHiddenDim() { return textHiddenDim; }
    public int fusionHiddenDim() { return fusionHiddenDim; }
    public int numFusionLayers() { return numFusionLayers; }
    public int numHeads() { return numHeads; }
    public int maxSeqLen() { return maxSeqLen; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("VisionLanguageModel(vocabSize=%d, imageHiddenDim=%d, textHiddenDim=%d, fusionHiddenDim=%d, numFusionLayers=%d, numHeads=%d, maxSeqLen=%d)",
                vocabSize, imageHiddenDim, textHiddenDim, fusionHiddenDim, numFusionLayers, numHeads, maxSeqLen);
    }
}
