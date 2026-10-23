package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.transformer.KVCache;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Token-by-token generation from a model mapping a token sequence to logits ({@code [seqLen, vocabSize]} values, the
 * last row scoring the next token). The logits processing of each step runs as {@link TensorOps} compositions on the
 * prompt's backend (device when the train kernels are available, CPU reference otherwise): selection of the last row,
 * repetition penalty ({@code x / penalty} for seen tokens with {@code x > 0}, else {@code x * penalty}), temperature
 * ({@code x / temperature}), greedy argmax and, for beam search, {@code logSoftmax}. Token sampling stays on the host:
 * greedy reads back one index; top-k, top-p and beam search read back the processed row once
 * per step and select on the host in {@link Float#compare} descending order with ties kept in index order. Every
 * generation API runs under {@link ComputationGraph#noGrad()}: nothing is recorded.
 */
public final class AutoregressiveGenerator implements AutoCloseable {
    private final Module model;
    private final int vocabSize;
    private final int dModel;
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int maxSeqLen;

    private KVCache kvCache;
    private volatile boolean closed = false;

    public AutoregressiveGenerator(Module model, int vocabSize, int dModel, int numLayers, int numHeads,
                                   int maxSeqLen) {
        Objects.requireNonNull(model, "Model cannot be null");
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be positive: " + vocabSize);
        }
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }

        this.model = model;
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.maxSeqLen = maxSeqLen;
    }

    /**
     * Extends the prompt until {@code min(maxLength, maxSeqLen)} tokens: top-k sampling when {@code topK > 0}, else
     * nucleus sampling when {@code topP < 1}, else greedy; the random stream is {@code new Random(config.seed())}.
     */
    public int[] generate(Tensor promptTokens, int maxLength, SamplingConfig config) {
        ensureOpen();
        Objects.requireNonNull(promptTokens, "Prompt tokens cannot be null");
        Objects.requireNonNull(config, "SamplingConfig cannot be null");

        if (maxLength <= 0) {
            throw new IllegalArgumentException("maxLength must be positive: " + maxLength);
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            ComputeBackend backend = promptTokens.backend();
            int[] tokens = readIntTokens(promptTokens);
            Random random = new Random(config.seed());
            int totalLength = Math.min(maxLength, maxSeqLen);

            while (tokens.length < totalLength) {
                int nextToken;
                try (Tensor processed = processedNextLogits(backend, tokens, config)) {
                    if (config.topK() > 0) {
                        nextToken = sampleTopK(processed.toFloatArray(), config.topK(), random);
                    } else if (config.topP() < 1.0f) {
                        nextToken = sampleTopP(processed.toFloatArray(), config.topP(), random);
                    } else {
                        nextToken = sampleGreedy(processed);
                    }
                }
                tokens = Arrays.copyOf(tokens, tokens.length + 1);
                tokens[tokens.length - 1] = nextToken;
            }
            return tokens;
        }
    }

    /**
     * {@link #generateBeamSearch} when {@code beamWidth > 1}, else {@link #generate(Tensor, int, SamplingConfig)} with
     * the prompt uploaded to the model's backend.
     *
     * @throws IllegalStateException when the model has no parameters to take a backend from
     */
    public int[] generate(int[] promptTokens, int maxLength, SamplingConfig config) {
        ensureOpen();
        Objects.requireNonNull(promptTokens, "Prompt tokens cannot be null");
        Objects.requireNonNull(config, "SamplingConfig cannot be null");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (config.beamWidth() > 1) {
                return generateBeamSearch(promptTokens, maxLength, config);
            }
            try (Tensor promptTensor = Tensor.fromArray(modelBackend(), promptTokens)) {
                return generate(promptTensor, maxLength, config);
            }
        }
    }

    /**
     * Beam search over {@code max(1, beamWidth)} beams scored by summed log-probabilities of the processed logits;
     * returns the best beam once it holds {@code min(maxLength, maxSeqLen)} tokens.
     *
     * @throws IllegalStateException when the model has no parameters to take a backend from
     */
    public int[] generateBeamSearch(int[] promptTokens, int maxLength, SamplingConfig config) {
        ensureOpen();
        Objects.requireNonNull(promptTokens, "Prompt tokens cannot be null");
        Objects.requireNonNull(config, "SamplingConfig cannot be null");

        if (maxLength <= 0) {
            throw new IllegalArgumentException("maxLength must be positive: " + maxLength);
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return beamSearch(promptTokens, maxLength, config);
        }
    }

    /** The search of {@link #generateBeamSearch}; the caller disables grad mode. */
    private int[] beamSearch(int[] promptTokens, int maxLength, SamplingConfig config) {
        ComputeBackend backend = modelBackend();
        int beamWidth = Math.max(1, config.beamWidth());

        List<BeamCandidate> beams = new ArrayList<>();
        beams.add(new BeamCandidate(Arrays.copyOf(promptTokens, promptTokens.length), 0.0f));

        int totalLength = Math.min(maxLength, maxSeqLen);

        while (beams.get(0).tokens.length < totalLength) {
            List<BeamCandidate> allCandidates = new ArrayList<>();

            for (BeamCandidate beam : beams) {
                if (beam.tokens.length >= maxSeqLen) {
                    allCandidates.add(beam);
                    continue;
                }

                float[] logProbs;
                try (Tensor processed = processedNextLogits(backend, beam.tokens, config);
                     Tensor logSoftmax = TensorOps.logSoftmax(processed, 0)) {
                    logProbs = logSoftmax.toFloatArray();
                }

                for (int token : topKIndices(logProbs, beamWidth)) {
                    int[] newTokens = Arrays.copyOf(beam.tokens, beam.tokens.length + 1);
                    newTokens[beam.tokens.length] = token;
                    allCandidates.add(new BeamCandidate(newTokens, beam.score + logProbs[token]));
                }
            }

            allCandidates.sort((a, b) -> Float.compare(b.score, a.score));

            beams.clear();
            for (int i = 0; i < Math.min(beamWidth, allCandidates.size()); i++) {
                beams.add(allCandidates.get(i));
            }
        }

        beams.sort((a, b) -> Float.compare(b.score, a.score));
        return beams.get(0).tokens;
    }

    /**
     * The model's last logits row for {@code tokens} after the repetition penalty and the temperature, as a new
     * {@code [vocabSize]} tensor owned by the caller.
     *
     * @throws IllegalArgumentException when the model output does not hold a whole number of {@code vocabSize} rows
     */
    private Tensor processedNextLogits(ComputeBackend backend, int[] tokens, SamplingConfig config) {
        Tensor lastRow;
        try (Tensor input = Tensor.fromArray(backend, tokens)) {
            Tensor logits = model.forward(input);
            try {
                lastRow = lastRow(logits);
            } finally {
                if (logits != input) {
                    logits.close();
                }
            }
        }
        try (Tensor row = lastRow;
             Tensor penalized = applyRepetitionPenalty(row, tokens, config.repetitionPenalty());
             Tensor temperature = constant(backend, config.temperature())) {
            return TensorOps.div(penalized, temperature);
        }
    }

    private Tensor lastRow(Tensor logits) {
        long count = logits.numElements();
        if (count == 0 || count % vocabSize != 0) {
            throw new IllegalArgumentException("Model output of shape " + logits.shape() + " does not hold whole rows"
                + " of " + vocabSize + " logits");
        }
        int rows = (int) (count / vocabSize);
        try (Tensor matrix = logits.reshape(rows, vocabSize);
             Tensor last = matrix.slice(0, rows - 1, rows)) {
            return last.reshape(vocabSize);
        }
    }

    /**
     * For every token of {@code prevTokens} inside the vocabulary: {@code x / penalty} when {@code x > 0}, else
     * {@code x * penalty}; a new tensor, or a new view of {@code row} when {@code penalty == 1}.
     */
    private Tensor applyRepetitionPenalty(Tensor row, int[] prevTokens, float penalty) {
        if (penalty == 1.0f) {
            return row.reshape(row.shape());
        }
        float[] seen = new float[vocabSize];
        for (int token : prevTokens) {
            if (token >= 0 && token < vocabSize) {
                seen[token] = 1.0f;
            }
        }
        ComputeBackend backend = row.backend();
        try (Tensor seenMask = Tensor.fromArray(backend, seen);
             Tensor zero = Tensor.zeros(backend, Shape.scalar());
             Tensor penaltyValue = constant(backend, penalty);
             Tensor positive = TensorOps.gt(row, zero);
             Tensor divided = TensorOps.div(row, penaltyValue);
             Tensor multiplied = TensorOps.mul(row, penalty);
             Tensor penalized = TensorOps.where(positive, divided, multiplied)) {
            return TensorOps.where(seenMask, penalized, row);
        }
    }

    /** The first index of the strict maximum (index 0 when none exceeds {@code -inf}), read back as one int. */
    private static int sampleGreedy(Tensor logits) {
        try (Tensor best = TensorOps.argmax(logits, 0)) {
            return best.toIntArray()[0];
        }
    }

    /**
     * Samples among the {@code k} highest logits ({@link #descendingOrder}) with probabilities
     * {@code exp(l - max) / sum} over those {@code k}, by the cumulative sum against one {@code nextFloat()}.
     */
    private static int sampleTopK(float[] logits, int k, Random random) {
        int actualK = Math.min(k, logits.length);
        Integer[] order = descendingOrder(logits);

        float[] topKProbs = new float[actualK];
        float maxLogit = logits[order[0]];
        float sumExp = 0.0f;
        for (int i = 0; i < actualK; i++) {
            topKProbs[i] = (float) Math.exp(logits[order[i]] - maxLogit);
            sumExp += topKProbs[i];
        }
        for (int i = 0; i < actualK; i++) {
            topKProbs[i] /= sumExp;
        }

        float r = random.nextFloat();
        float cumProb = 0.0f;
        for (int i = 0; i < actualK; i++) {
            cumProb += topKProbs[i];
            if (r <= cumProb) {
                return order[i];
            }
        }
        return order[actualK - 1];
    }

    /**
     * Nucleus sampling: probabilities {@code exp(l - max) / sum} in {@link #descendingOrder}, cut at the first index
     * whose cumulative probability reaches {@code p}, then sampled against {@code nextFloat() * p}.
     */
    private static int sampleTopP(float[] logits, float p, Random random) {
        Integer[] order = descendingOrder(logits);

        float maxLogit = logits[order[0]];
        float sumExp = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[order[i]] - maxLogit);
            sumExp += probs[i];
        }

        float cumProb = 0.0f;
        int cutoffIdx = 0;
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
            cumProb += probs[i];
            if (cumProb >= p) {
                cutoffIdx = i;
                break;
            }
        }

        float r = random.nextFloat() * p;
        cumProb = 0.0f;
        for (int i = 0; i <= cutoffIdx; i++) {
            cumProb += probs[i];
            if (r <= cumProb) {
                return order[i];
            }
        }
        return order[cutoffIdx];
    }

    /** The indices of the {@code min(k, length)} highest scores in {@link #descendingOrder}. */
    private static int[] topKIndices(float[] scores, int k) {
        Integer[] order = descendingOrder(scores);
        int[] result = new int[Math.min(k, order.length)];
        for (int i = 0; i < result.length; i++) {
            result[i] = order[i];
        }
        return result;
    }

    /** All indices sorted by {@link Float#compare} on their values, descending; ties stay in index order. */
    private static Integer[] descendingOrder(float[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(values[b], values[a]));
        return order;
    }

    private static final class BeamCandidate {
        final int[] tokens;
        final float score;

        BeamCandidate(int[] tokens, float score) {
            this.tokens = tokens;
            this.score = score;
        }
    }

    /** The prompt as Java ints: INT32 as is, INT64 narrowed by {@code (int)}, other dtypes rounded. */
    private static int[] readIntTokens(Tensor tensor) {
        if (tensor.dtype() == DType.INT32 || tensor.dtype() == DType.INT64) {
            return tensor.toIntArray();
        }
        float[] floatData = tensor.toFloatArray();
        int[] tokens = new int[floatData.length];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = Math.round(floatData[i]);
        }
        return tokens;
    }

    /** A rank-0 tensor holding {@code value}, filled on the device when the train kernels are available. */
    private static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    private ComputeBackend modelBackend() {
        ComputeBackend backend = model.backend();
        if (backend == null) {
            throw new IllegalStateException("The model has no parameters to take a backend from; pass the prompt as"
                + " a tensor on the model's backend");
        }
        return backend;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("AutoregressiveGenerator has been closed");
        }
    }

    public Module model() {
        return model;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int dModel() {
        return dModel;
    }

    public int numLayers() {
        return numLayers;
    }

    public int numHeads() {
        return numHeads;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (kvCache != null) {
                kvCache.close();
                kvCache = null;
            }
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "AutoregressiveGenerator{closed=true}";
        }
        return "AutoregressiveGenerator{" +
            "vocabSize=" + vocabSize +
            ", dModel=" + dModel +
            ", numLayers=" + numLayers +
            ", numHeads=" + numHeads +
            ", maxSeqLen=" + maxSeqLen +
            '}';
    }
}
