package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Fixed sinusoidal position encoding: {@link #initialize} uploads the table {@code [maxSeqLen, dModel]} once, and
 * {@link #apply} adds rows {@code [offset, offset + seqLen)} to the input on its backend, broadcast over the batch
 * ({@code input + pe}, one rounding per element as in Java float arithmetic).
 *
 * <p>The table is a constant; in grad mode the addition records and the input receives the upstream gradient.</p>
 */
public final class SinusoidalPE implements PositionalEncoding {
    private final int dModel;
    private final int maxSeqLen;
    private final float dropoutRate;
    private Tensor peBuffer;

    public SinusoidalPE(int dModel, int maxSeqLen) {
        this(dModel, maxSeqLen, 0.0f);
    }

    public SinusoidalPE(int dModel, int maxSeqLen, float dropoutRate) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException("dropoutRate must be in [0.0, 1.0): " + dropoutRate);
        }
        
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.dropoutRate = dropoutRate;
    }

    public void initialize(com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        
        float[] peData = new float[maxSeqLen * dModel];
        
        float divTerm;
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dModel; i++) {
                int idx = pos * dModel + i;
                if (i % 2 == 0) {
                    divTerm = (float) Math.exp(-Math.log(10000.0) * (i / (float) dModel));
                    peData[idx] = (float) Math.sin(pos * divTerm);
                } else {
                    divTerm = (float) Math.exp(-Math.log(10000.0) * ((i - 1) / (float) dModel));
                    peData[idx] = (float) Math.cos(pos * divTerm);
                }
            }
        }
        
        close();
        try (Tensor flat = Tensor.fromArray(backend, peData)) {
            peBuffer = flat.reshape(Shape.of(maxSeqLen, dModel));
        }
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
        
        try (Tensor pe = peBuffer.slice(0, offset, offset + seqLen)) {
            return TensorOps.add(input, pe);
        }
    }

    private void ensureInitialized() {
        if (peBuffer == null) {
            throw new IllegalStateException("SinusoidalPE not initialized. Call initialize(backend) first.");
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
        return "SinusoidalPE";
    }

    public float dropoutRate() {
        return dropoutRate;
    }

    public Tensor getPEBuffer() {
        ensureInitialized();
        return peBuffer;
    }

    public void close() {
        if (peBuffer != null) {
            peBuffer.close();
            peBuffer = null;
        }
    }

    @Override
    public String toString() {
        return "SinusoidalPE{" +
            "dModel=" + dModel +
            ", maxSeqLen=" + maxSeqLen +
            ", dropoutRate=" + dropoutRate +
            ", initialized=" + (peBuffer != null) +
            '}';
    }
}
