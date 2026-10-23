package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

public final class Embedding extends Module {
    private final int numEmbeddings;
    private final int embeddingDim;
    private final int paddingIdx;
    
    private Parameter weight;

    public Embedding(int numEmbeddings, int embeddingDim) {
        this(numEmbeddings, embeddingDim, -1);
    }

    public Embedding(int numEmbeddings, int embeddingDim, int paddingIdx) {
        if (numEmbeddings <= 0) {
            throw new IllegalArgumentException("numEmbeddings must be positive: " + numEmbeddings);
        }
        if (embeddingDim <= 0) {
            throw new IllegalArgumentException("embeddingDim must be positive: " + embeddingDim);
        }
        if (paddingIdx >= numEmbeddings) {
            throw new IllegalArgumentException("paddingIdx must be less than numEmbeddings: " + paddingIdx);
        }
        
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIdx = paddingIdx;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Registers the {@code [numEmbeddings, embeddingDim]} table drawn from N(0, 1 / embeddingDim) on the host, with the
     * padding row zeroed when there is one. Runs under {@link ComputationGraph#noGrad()}; the parameter owns the
     * tensor (an autograd leaf).
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        float std = (float) (1.0 / Math.sqrt(embeddingDim));
        
        float[] weightData = new float[numEmbeddings * embeddingDim];
        for (int i = 0; i < weightData.length; i++) {
            weightData[i] = (float) (random.nextGaussian() * std);
        }
        
        if (paddingIdx >= 0) {
            for (int d = 0; d < embeddingDim; d++) {
                weightData[paddingIdx * embeddingDim + d] = 0.0f;
            }
        }
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor table = Tensor.fromStorage(backend, Shape.of(numEmbeddings, embeddingDim), DType.FLOAT32,
                TensorStorage.fromHostFloat(backend, weightData));
            this.weight = registerParameter("weight", new Parameter(table, "weight"));
        }
    }

    /**
     * Rows of the embedding table selected by {@code indices}: output shape {@code indices.shape + [embeddingDim]},
     * in the table's dtype and on the table's backend, computed by
     * {@link TensorOps#indexSelectRows(Tensor, Tensor, boolean, int)} in error mode. In grad mode the op records
     * {@code IndexSelectRowsGrad}: the table gradient sums the output gradient rows of each selected index, and the
     * padding row (when {@code paddingIdx >= 0}) gets no gradient, as in PyTorch.
     *
     * <p>Indices on another backend than the table are copied to the table's backend first (an integer transfer that
     * carries no gradient). On a device backend with the train library, an index outside {@code [0, numEmbeddings)}
     * surfaces as {@link IndexOutOfBoundsException} at the backend's next host synchronization; on the CPU reference
     * path it throws immediately.</p>
     *
     * @param indices INT32 or INT64 indices of any shape; INT64 values are narrowed with Java {@code (int)}
     * @throws IllegalArgumentException  when the indices are not INT32 or INT64
     * @throws IndexOutOfBoundsException when an index is outside {@code [0, numEmbeddings)}
     */
    public Tensor forward(Tensor indices) {
        ensureInitialized();
        Objects.requireNonNull(indices, "Indices cannot be null");
        
        if (indices.dtype() != DType.INT32 && indices.dtype() != DType.INT64) {
            throw new IllegalArgumentException("Indices tensor must be INT32 or INT64, got " + indices.dtype());
        }
        
        Tensor table = weight.tensor();
        if (indices.backend() == table.backend()) {
            return TensorOps.indexSelectRows(table, indices, false, paddingIdx);
        }
        try (Tensor flat = Tensor.fromArray(table.backend(), indices.toIntArray());
             Tensor local = flat.reshape(indices.shape())) {
            return TensorOps.indexSelectRows(table, local, false, paddingIdx);
        }
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("Embedding layer not initialized. Call initialize(backend) first.");
        }
    }

    public int numEmbeddings() {
        return numEmbeddings;
    }

    public int embeddingDim() {
        return embeddingDim;
    }

    public int paddingIdx() {
        return paddingIdx;
    }

    public Parameter weight() {
        ensureInitialized();
        return weight;
    }

    public Shape computeOutputShape(Shape inputShape) {
        int[] outDims = new int[inputShape.rank() + 1];
        for (int i = 0; i < inputShape.rank(); i++) {
            outDims[i] = inputShape.dim(i);
        }
        outDims[inputShape.rank()] = embeddingDim;
        return Shape.of(outDims);
    }

    @Override
    public String toString() {
        return "Embedding{" +
            "numEmbeddings=" + numEmbeddings +
            ", embeddingDim=" + embeddingDim +
            ", paddingIdx=" + paddingIdx +
            ", initialized=" + (weight != null) +
            '}';
    }
}
