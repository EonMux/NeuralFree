package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/**
 * Routes an input to one of several modalities: a linear layer over the (pooled) input gives per-modality logits,
 * turned into weights by softmax or sigmoid at a temperature, and the selection is the top weight (or the first of the
 * top-k). The weights are computed on the input's backend; the routing decision is read back to Java.
 *
 * <p><b>Training.</b> {@link #route} is the inference path: it runs without recording and returns Java values. The
 * training path is {@link #routeWeights}, which returns the recorded weight tensor, and
 * {@link #combineEmbeddings(Map, Tensor)}, which mixes the embeddings with it, so a loss on the combined embedding
 * reaches the routing layer. The top-k selection is a constant mask read from the weights; the weights of the
 * selected modalities keep their gradient.</p>
 */
public class ModalityRouter extends Module {

    private final int hiddenDim;
    private final int numModalities;
    private final String routingStrategy;
    private final float routingTemperature;
    private final boolean useTopK;
    private final int topK;

    private Linear routingLayer;
    private final Map<String, Integer> modalityToIndex;
    private final List<String> modalityNames;

    private boolean initialized = false;

    public static final String MODALITY_VISION = "vision";
    public static final String MODALITY_TEXT = "text";
    public static final String MODALITY_AUDIO = "audio";

    public ModalityRouter(int hiddenDim, int numModalities, String routingStrategy,
                          float routingTemperature, boolean useTopK, int topK) {
        if (hiddenDim <= 0) throw new IllegalArgumentException("hiddenDim must be positive");
        if (numModalities <= 0) throw new IllegalArgumentException("numModalities must be positive");
        if (routingTemperature <= 0) throw new IllegalArgumentException("routingTemperature must be positive");
        if (useTopK && (topK <= 0 || topK > numModalities)) {
            throw new IllegalArgumentException("topK must be in [1, numModalities] when useTopK is true");
        }

        this.hiddenDim = hiddenDim;
        this.numModalities = numModalities;
        this.routingStrategy = routingStrategy != null ? routingStrategy : "softmax";
        this.routingTemperature = routingTemperature;
        this.useTopK = useTopK;
        this.topK = topK;

        this.modalityToIndex = new HashMap<>();
        this.modalityNames = new ArrayList<>();
    }

    public ModalityRouter(int hiddenDim, int numModalities) {
        this(hiddenDim, numModalities, "softmax", 1.0f, false, 1);
    }

    public void initialize(ComputeBackend backend, String... modalities) {
        initialize(backend, new Random(), modalities);
    }

    /**
     * Registers the modality names (vision, text and audio when none are given) and creates the routing layer on
     * {@code backend}, drawing its weights from {@code random} (a new one when null).
     *
     * @throws IllegalArgumentException when the number of names differs from {@code numModalities}
     */
    public void initialize(ComputeBackend backend, Random random, String... modalities) {
        if (initialized) return;

        if (modalities == null || modalities.length == 0) {
            modalities = new String[]{MODALITY_VISION, MODALITY_TEXT, MODALITY_AUDIO};
        }

        if (modalities.length != numModalities) {
            throw new IllegalArgumentException("Number of modalities must match numModalities");
        }

        for (int i = 0; i < modalities.length; i++) {
            modalityToIndex.put(modalities[i], i);
            modalityNames.add(modalities[i]);
        }

        routingLayer = new Linear(hiddenDim, numModalities);
        routingLayer.initialize(backend, random != null ? random : new Random());
        registerModule("routingLayer", routingLayer);

        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("ModalityRouter.forward(Tensor) is not supported. Use route(Tensor) method to get routing results.");
    }

    /**
     * Routes a {@code [batch, hiddenDim]} input, or a higher-rank input pooled to it (the mean over dim 1 for rank 3;
     * means over the trailing dims, last first, for rank 4 and above). The routing weights are computed on the input's
     * backend ({@code softmax} or {@code sigmoid} of {@code logits / routingTemperature}, or the raw logits for any
     * other strategy) and read back once: the weights and the per-row selection are Java values the caller's control
     * flow consumes. Runs under {@link ComputationGraph#noGrad()} and records nothing; training
     * uses {@link #routeWeights}.
     *
     * @throws IllegalArgumentException when the input has rank below 2
     */
    public RoutingResult route(Tensor input) {
        requireRoutable(input);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor weightTensor = weightsOf(input)) {
            int batchSize = weightTensor.shape().dim(0);
            float[] weights = weightTensor.toFloatArray();
            int[] selectedModalities = selectModalities(weights, batchSize);
            return new RoutingResult(weights, selectedModalities, modalityNames);
        }
    }

    /**
     * The routing weights of {@link #route} as a new {@code [batch, numModalities]} tensor owned by the caller, from
     * the same pooling, routing layer and strategy, recorded when the routing layer or the input requires grad. With
     * top-k routing, the entries outside each row's top {@code topK} (selected as {@link #route} selects them, on
     * values read back once) are replaced by 0 through {@code where} with a constant mask; the selected entries
     * keep their values and gradients. Without top-k every entry is kept.
     *
     * @throws IllegalArgumentException when the input has rank below 2
     */
    public Tensor routeWeights(Tensor input) {
        requireRoutable(input);
        Tensor weights = weightsOf(input);
        if (!useTopK) {
            return weights;
        }
        try (Tensor all = weights) {
            return keepTopK(all);
        }
    }

    /**
     * {@link #combineEmbeddings(Map, Tensor)} with host weights: {@code weights} holds either one row of
     * {@code numModalities} values, applied to every embedding row, or {@code batch} rows in row-major
     * {@code [batch, numModalities]} order (as {@link RoutingResult#weights()} of a routed batch), row {@code b}
     * weighting row {@code b} of every embedding. The weights become a constant {@code [batch, numModalities]} tensor
     * on the embeddings' backend, so no gradient reaches the router through them; the embeddings receive theirs.
     * Training uses {@link #combineEmbeddings(Map, Tensor)} with {@link #routeWeights}. Returns null when no
     * registered modality is present; a row without a positive weight is 0.
     *
     * @throws IllegalArgumentException when {@code embeddings} is null or empty, {@code weights} holds neither
     *                                  {@code numModalities} nor {@code batch * numModalities} values or a value that
     *                                  is not finite, or the embeddings are not {@code [batch, ...]} of one shape
     */
    public Tensor combineEmbeddings(Map<String, Tensor> embeddings, float[] weights) {
        if (embeddings == null || embeddings.isEmpty()) {
            throw new IllegalArgumentException("embeddings map cannot be null or empty");
        }
        Objects.requireNonNull(weights, "weights cannot be null");
        for (int i = 0; i < weights.length; i++) {
            if (!Float.isFinite(weights[i])) {
                throw new IllegalArgumentException("weights must be finite, got " + weights[i] + " at index " + i);
            }
        }
        Tensor first = firstRegisteredEmbedding(embeddings);
        if (first == null) {
            return null;
        }
        if (first.shape().rank() < 1) {
            throw new IllegalArgumentException("embeddings must be [batch, ...], got " + first.shape());
        }
        int batchSize = first.shape().dim(0);
        float[] rows = weightRows(weights, batchSize);
        try (Tensor flat = Tensor.fromArray(first.backend(), rows);
             Tensor weightTensor = flat.reshape(Shape.matrix(batchSize, numModalities))) {
            return combineEmbeddings(embeddings, weightTensor);
        }
    }

    /** The embedding of the first registered modality in map order, or null when there is none. */
    private Tensor firstRegisteredEmbedding(Map<String, Tensor> embeddings) {
        for (Map.Entry<String, Tensor> entry : embeddings.entrySet()) {
            if (modalityToIndex.containsKey(entry.getKey())) {
                return Objects.requireNonNull(entry.getValue(), "embedding of " + entry.getKey() + " cannot be null");
            }
        }
        return null;
    }

    /** The row-major {@code [batchSize, numModalities]} weights: {@code weights} itself or its one row repeated. */
    private float[] weightRows(float[] weights, int batchSize) {
        long perBatch = (long) batchSize * numModalities;
        if (weights.length == perBatch) {
            return weights.clone();
        }
        if (weights.length != numModalities) {
            throw new IllegalArgumentException("weights must hold " + numModalities + " values (one row for every "
                    + "embedding row) or " + perBatch + " values (row-major [" + batchSize + ", " + numModalities
                    + "]), got " + weights.length);
        }
        float[] rows = new float[Math.toIntExact(perBatch)];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(weights, 0, rows, b * numModalities, numModalities);
        }
        return rows;
    }

    /**
     * The per-row weighted mean of the registered modalities in {@code embeddings} under the
     * {@code [batch, numModalities]} {@code weights} (typically {@link #routeWeights}), as a new tensor of the
     * embeddings' shape: row {@code b} is {@code sum_m w+_bm * e_m[b] / sum_m w+_bm} with {@code w+ = w} where
     * {@code w > 0} and 0 elsewhere, the positive-weight rule of the float overload applied per row; a row without a
     * positive weight is 0. Every embedding has the shape {@code [batch, ...]}; unregistered names are skipped, and
     * the result is null when no registered modality is present. The weights are sliced and multiplied as recorded
     * operations, so the gradient of a loss on the result reaches the weights (and through {@link #routeWeights} the
     * routing layer) and the embeddings; the {@code w > 0} masks are constants, and the zero-row divisor is replaced
     * by 1 so that row's gradient is 0.
     *
     * @throws IllegalArgumentException when the weights are not {@code [batch, numModalities]}, or a registered
     *                                  embedding is not {@code [batch, ...]} or differs in shape from another
     */
    public Tensor combineEmbeddings(Map<String, Tensor> embeddings, Tensor weights) {
        if (embeddings == null || embeddings.isEmpty()) {
            throw new IllegalArgumentException("embeddings map cannot be null or empty");
        }
        Objects.requireNonNull(weights, "weights cannot be null");
        if (weights.shape().rank() != 2 || weights.shape().dim(1) != numModalities) {
            throw new IllegalArgumentException("weights must be [batch, " + numModalities + "], got "
                    + weights.shape());
        }
        int batchSize = weights.shape().dim(0);

        Tensor numerator = null;
        Tensor denominator = null;
        Shape embeddingShape = null;
        try {
            for (Map.Entry<String, Tensor> entry : embeddings.entrySet()) {
                Integer modalityIdx = modalityToIndex.get(entry.getKey());
                if (modalityIdx == null) {
                    continue;
                }
                Tensor embedding = Objects.requireNonNull(entry.getValue(),
                        "embedding of " + entry.getKey() + " cannot be null");
                embeddingShape = requireEmbeddingShape(entry.getKey(), embedding, embeddingShape, batchSize);
                Tensor positive;
                try (Tensor column = weights.slice(1, modalityIdx, modalityIdx + 1)) {
                    positive = positivePart(column);
                }
                Tensor weighted;
                try (Tensor scale = positive.reshape(rowBroadcastShape(embeddingShape))) {
                    weighted = TensorOps.mul(embedding, scale);
                } catch (RuntimeException e) {
                    positive.close();
                    throw e;
                }
                denominator = accumulate(denominator, positive, weighted);
                numerator = accumulate(numerator, weighted, null);
            }
            if (numerator == null) {
                return null;
            }
            try (Tensor sum = numerator;
                 Tensor total = denominator;
                 Tensor divisor = nonZeroDivisor(total);
                 Tensor scale = divisor.reshape(rowBroadcastShape(embeddingShape))) {
                numerator = null;
                denominator = null;
                return TensorOps.div(sum, scale);
            }
        } catch (RuntimeException e) {
            if (numerator != null) {
                numerator.close();
            }
            if (denominator != null) {
                denominator.close();
            }
            throw e;
        }
    }

    private void requireRoutable(Tensor input) {
        if (!initialized) {
            throw new IllegalStateException("ModalityRouter not initialized. Call initialize() first.");
        }
        if (input == null) {
            throw new IllegalArgumentException("input tensor cannot be null");
        }
        if (input.shape().rank() < 2) {
            throw new IllegalArgumentException("input must have a batch and a feature dim, got " + input.shape());
        }
    }

    /** The strategy's {@code [batch, numModalities]} weights of {@code input} as a new tensor. */
    private Tensor weightsOf(Tensor input) {
        Tensor pooled = poolInput(input);
        try {
            Tensor logits = routingLayer.forward(pooled);
            Tensor weights;
            try {
                weights = routingWeights(logits);
            } catch (RuntimeException e) {
                logits.close();
                throw e;
            }
            if (weights == null) {
                return logits;
            }
            logits.close();
            return weights;
        } finally {
            if (pooled != input) {
                pooled.close();
            }
        }
    }

    /** {@code weights} with every entry outside its row's top {@code topK} replaced by 0, as a new tensor. */
    private Tensor keepTopK(Tensor weights) {
        int batchSize = weights.shape().dim(0);
        float[] values = weights.toFloatArray();
        float[] keep = new float[values.length];
        for (int b = 0; b < batchSize; b++) {
            int offset = b * numModalities;
            for (int index : findTopK(values, offset, topK)) {
                keep[offset + index] = 1.0f;
            }
        }
        ComputeBackend backend = weights.backend();
        try (Tensor flatMask = Tensor.fromArray(backend, keep);
             Tensor mask = flatMask.reshape(weights.shape());
             Tensor zeros = Tensor.zeros(backend, weights.shape())) {
            return TensorOps.where(mask, weights, zeros);
        }
    }

    /** {@code w} where {@code w > 0}, else 0, as a new tensor (the comparison is a constant mask). */
    private static Tensor positivePart(Tensor w) {
        try (Tensor zeros = Tensor.zeros(w.backend(), w.shape());
             Tensor positive = TensorOps.gt(w, zeros)) {
            return TensorOps.where(positive, w, zeros);
        }
    }

    /** {@code total} where it is positive, else 1, as a new tensor. */
    private static Tensor nonZeroDivisor(Tensor total) {
        try (Tensor zeros = Tensor.zeros(total.backend(), total.shape());
             Tensor ones = Tensor.ones(total.backend(), total.shape());
             Tensor positive = TensorOps.gt(total, zeros)) {
            return TensorOps.where(positive, total, ones);
        }
    }

    /**
     * {@code acc + term} as a new tensor (closing both), or {@code term} itself when {@code acc} is null; on failure
     * {@code acc}, {@code term} and {@code alsoClose} (when not null) are closed.
     */
    private static Tensor accumulate(Tensor acc, Tensor term, Tensor alsoClose) {
        if (acc == null) {
            return term;
        }
        try (Tensor previous = acc; Tensor next = term) {
            return TensorOps.add(previous, next);
        } catch (RuntimeException e) {
            if (alsoClose != null) {
                alsoClose.close();
            }
            throw e;
        }
    }

    /** {@code [batch, 1, ..., 1]} of the embedding's rank, to broadcast a per-row weight over it. */
    private static Shape rowBroadcastShape(Shape embeddingShape) {
        int[] dims = new int[embeddingShape.rank()];
        Arrays.fill(dims, 1);
        dims[0] = embeddingShape.dim(0);
        return Shape.of(dims);
    }

    private static Shape requireEmbeddingShape(String name, Tensor embedding, Shape expected, int batchSize) {
        Shape shape = embedding.shape();
        if (shape.rank() < 1 || shape.dim(0) != batchSize) {
            throw new IllegalArgumentException("embedding of " + name + " must be [" + batchSize + ", ...], got "
                    + shape);
        }
        if (expected != null && !expected.equals(shape)) {
            throw new IllegalArgumentException("embedding of " + name + " has shape " + shape + ", the others "
                    + expected);
        }
        return shape;
    }

    private Tensor poolInput(Tensor input) {
        int rank = input.shape().rank();

        if (rank == 3) {
            return TensorOps.mean(input, 1);
        }
        Tensor result = input;
        for (int i = rank - 1; i > 1; i--) {
            Tensor reduced = TensorOps.mean(result, i);
            if (result != input) {
                result.close();
            }
            result = reduced;
        }
        return result;
    }

    /** The routing weights of the strategy as a new tensor, or null when the strategy uses the logits as they are. */
    private Tensor routingWeights(Tensor logits) {
        boolean softmax = "softmax".equals(routingStrategy);
        if (!softmax && !"sigmoid".equals(routingStrategy)) {
            return null;
        }
        try (Tensor temperature = Tensor.scalar(logits.backend(), routingTemperature);
             Tensor scaled = TensorOps.div(logits, temperature)) {
            return softmax ? TensorOps.softmax(scaled, 1) : TensorOps.sigmoid(scaled);
        }
    }

    private int[] selectModalities(float[] weights, int batchSize) {
        int[] selected = new int[batchSize];

        for (int b = 0; b < batchSize; b++) {
            int offset = b * numModalities;

            if (useTopK) {
                int[] topIndices = findTopK(weights, offset, topK);
                selected[b] = topIndices[0];
            } else {
                int maxIdx = 0;
                float maxWeight = weights[offset];
                for (int m = 1; m < numModalities; m++) {
                    if (weights[offset + m] > maxWeight) {
                        maxWeight = weights[offset + m];
                        maxIdx = m;
                    }
                }
                selected[b] = maxIdx;
            }
        }

        return selected;
    }

    private int[] findTopK(float[] weights, int offset, int k) {
        int[] indices = new int[k];
        float[] topWeights = new float[k];
        for (int i = 0; i < k; i++) {
            topWeights[i] = Float.NEGATIVE_INFINITY;
        }

        for (int m = 0; m < numModalities; m++) {
            float w = weights[offset + m];
            for (int i = 0; i < k; i++) {
                if (w > topWeights[i]) {
                    for (int j = k - 1; j > i; j--) {
                        topWeights[j] = topWeights[j - 1];
                        indices[j] = indices[j - 1];
                    }
                    topWeights[i] = w;
                    indices[i] = m;
                    break;
                }
            }
        }

        return indices;
    }

    public int getModalityIndex(String modality) {
        if (!modalityToIndex.containsKey(modality)) {
            throw new IllegalArgumentException("Unknown modality: " + modality);
        }
        return modalityToIndex.get(modality);
    }

    public String getModalityName(int index) {
        if (index < 0 || index >= numModalities) {
            throw new IllegalArgumentException("Invalid modality index: " + index);
        }
        return modalityNames.get(index);
    }

    public Set<String> getRegisteredModalities() {
        return modalityToIndex.keySet();
    }

    public int hiddenDim() { return hiddenDim; }
    public int numModalities() { return numModalities; }
    public String routingStrategy() { return routingStrategy; }
    public float routingTemperature() { return routingTemperature; }
    public boolean useTopK() { return useTopK; }
    public int topK() { return topK; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("ModalityRouter(hiddenDim=%d, numModalities=%d, routingStrategy=%s, routingTemperature=%.2f, useTopK=%s, topK=%d)",
                hiddenDim, numModalities, routingStrategy, routingTemperature, useTopK, topK);
    }

    public static class RoutingResult {
        private final float[] weights;
        private final int[] selectedModalities;
        private final List<String> modalityNames;

        public RoutingResult(float[] weights, int[] selectedModalities, List<String> modalityNames) {
            this.weights = weights;
            this.selectedModalities = selectedModalities;
            this.modalityNames = modalityNames;
        }

        public float[] weights() { return weights; }
        public int[] selectedModalities() { return selectedModalities; }
        public List<String> modalityNames() { return modalityNames; }

        public float getWeight(int batchIndex, int modalityIndex) {
            return weights[batchIndex * modalityNames.size() + modalityIndex];
        }

        public String getSelectedModalityName(int batchIndex) {
            return modalityNames.get(selectedModalities[batchIndex]);
        }

        public int getSelectedModalityIndex(int batchIndex) {
            return selectedModalities[batchIndex];
        }
    }
}
