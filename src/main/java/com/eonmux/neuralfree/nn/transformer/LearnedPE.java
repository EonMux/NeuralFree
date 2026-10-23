package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;
import java.util.Random;

/**
 * Learned position embedding: a trainable table {@code [maxSeqLen, dModel]} ({@code pe_embedding}) whose rows
 * {@code [offset, offset + seqLen)} {@link #apply} adds to the input on its backend, broadcast over the batch
 * ({@code input + pe}, one rounding per element as in Java float arithmetic).
 *
 * <p>Each call slices the parameter's current tensor ({@link Parameter#tensor()}, so a substitution is honoured) and
 * caches nothing: in grad mode the slice and the addition record, and the backward pass delivers to
 * {@code pe_embedding} the gradient summed over the batch at rows {@code [offset, offset + seqLen)} and zero
 * elsewhere.</p>
 */
public final class LearnedPE extends Module implements PositionalEncoding {
    private final int dModel;
    private final int maxSeqLen;
    private Parameter peEmbedding;
    private boolean initialized = false;

    public LearnedPE(int dModel, int maxSeqLen) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Creates the trainable table {@code pe_embedding} on {@code backend}, each entry
     * {@code nextGaussian() * (1 / sqrt(dModel))} from {@code random} in row-major order. Runs under
     * {@link ComputationGraph#noGrad()}; the {@link Parameter} makes the table a leaf.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        float[] peData = new float[maxSeqLen * dModel];
        float std = (float) (1.0 / Math.sqrt(dModel));
        for (int i = 0; i < peData.length; i++) {
            peData[i] = (float) (random.nextGaussian() * std);
        }
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor table;
            try (Tensor flat = Tensor.fromArray(backend, peData)) {
                table = flat.reshape(Shape.of(maxSeqLen, dModel));
            }
            this.peEmbedding = registerParameter("pe_embedding", new Parameter(table, "pe_embedding"));
        }
        this.initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        return apply(input, 0);
    }

    @Override
    public Tensor apply(Tensor input, int offset) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape shape = input.shape();
        int rank = shape.rank();
        
        if (rank != 2 && rank != 3) {
            throw new IllegalArgumentException("Input must be 2D [seq_len, d_model] or 3D [batch, seq_len, d_model], got rank " + rank);
        }
        
        int seqLen = shape.dim(rank - 2);
        int dModelInput = shape.dim(rank - 1);
        
        if (dModelInput != dModel) {
            throw new IllegalArgumentException("Input d_model (" + dModelInput + ") does not match PE d_model (" + dModel + ")");
        }
        
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be non-negative: " + offset);
        }
        if (seqLen == 0) {
            throw new IllegalArgumentException("Input sequence length must be positive, got shape " + shape);
        }
        if (offset + seqLen > maxSeqLen) {
            throw new IllegalArgumentException("Position " + (offset + seqLen - 1) + " exceeds maxSeqLen " + maxSeqLen);
        }
        
        try (Tensor pe = peEmbedding.tensor().slice(0, offset, offset + seqLen)) {
            return TensorOps.add(input, pe);
        }
    }

    private void ensureInitialized() {
        if (!initialized || peEmbedding == null) {
            throw new IllegalStateException("LearnedPE not initialized. Call initialize(backend) first.");
        }
    }

    @Override
    public int maxSequenceLength() {
        return maxSeqLen;
    }

    @Override
    public int dModel() {
        return dModel;
    }

    @Override
    public String name() {
        return "LearnedPE";
    }

    public Parameter getPEEmbedding() {
        ensureInitialized();
        return peEmbedding;
    }

    @Override
    public String toString() {
        return "LearnedPE{" +
            "dModel=" + dModel +
            ", maxSeqLen=" + maxSeqLen +
            ", initialized=" + initialized +
            '}';
    }
}
