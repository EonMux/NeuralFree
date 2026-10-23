package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Attention with linear biases (Press et al.): {@link #apply} leaves the input unchanged, and
 * {@link #getAttentionBias} returns the per-head bias {@code -slope[h] * |j - (offset + i)|}
 * {@code [numHeads, queryLen, keyLen]} to add to the attention scores (for example as the mask of
 * {@link MultiHeadAttention#forward(Tensor, Tensor, Tensor, Tensor)}), so each key is penalized in proportion to its
 * distance from the query. Under a causal mask this is the paper's {@code slope[h] * (j - i)}; without one it is the
 * symmetric form llama.cpp applies. The head slopes are the reference geometric sequence with maximum bias 8
 * ({@link CpuAttentionCore#alibiSlope}): with {@code p} the largest power of two not above {@code n} heads, head
 * {@code h < p} has slope {@code 2^(-8(h+1)/p)} and each extra head {@code h >= p} has {@code 2^(-4(2(h-p)+1)/p)}.
 * The bias table and every bias returned are constants that never require grad.
 */
public final class ALiBiPE implements PositionalEncoding {
    /** The maximum bias of the reference slope sequence: the first head of a power-of-two count has slope 2^(-8/n). */
    private static final float MAX_BIAS = 8.0f;

    private final int dModel;
    private final int maxSeqLen;
    private final int numHeads;
    private final float[] slopes;
    private Tensor biasCache;

    public ALiBiPE(int dModel, int maxSeqLen, int numHeads) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.numHeads = numHeads;
        this.slopes = computeSlopes(numHeads);
    }

    private static float[] computeSlopes(int numHeads) {
        float[] slopes = new float[numHeads];
        for (int h = 0; h < numHeads; h++) {
            slopes[h] = CpuAttentionCore.alibiSlope(h, numHeads, MAX_BIAS);
        }
        return slopes;
    }

    /**
     * Builds the bias table {@code [numHeads, maxSeqLen, maxSeqLen]} on {@code backend} as a broadcast product
     * {@code bias[h, i, j] = -slope[h] * |j - i|}: the per-head negated slopes {@code [numHeads, 1, 1]} times the
     * distance matrix {@code |cols - rows|}, whose integer values are exact in float, so each entry is the single
     * rounding of {@code -slope * |j - i|}.
     */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        
        float[] negSlopes = new float[numHeads];
        for (int h = 0; h < numHeads; h++) {
            negSlopes[h] = -slopes[h];
        }
        float[] positions = new float[maxSeqLen];
        for (int i = 0; i < maxSeqLen; i++) {
            positions[i] = i;
        }
        
        close();
        try (Tensor slopeFlat = Tensor.fromArray(backend, negSlopes);
             Tensor slopeCol = slopeFlat.reshape(Shape.of(numHeads, 1, 1));
             Tensor pos = Tensor.fromArray(backend, positions);
             Tensor rows = pos.reshape(Shape.of(maxSeqLen, 1));
             Tensor cols = pos.reshape(Shape.of(1, maxSeqLen));
             Tensor offsets = TensorOps.sub(cols, rows);
             Tensor distance = TensorOps.abs(offsets)) {
            biasCache = TensorOps.mul(slopeCol, distance);
        }
    }

    @Override
    public Tensor apply(Tensor input, int offset) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        return input;
    }

    public Tensor getAttentionBias(int queryLen, int keyLen) {
        return getAttentionBias(queryLen, keyLen, 0);
    }

    public Tensor getAttentionBias(int queryLen, int keyLen, int offset) {
        ensureInitialized();
        
        if (queryLen <= 0 || keyLen <= 0) {
            throw new IllegalArgumentException("queryLen and keyLen must be positive");
        }
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be non-negative: " + offset);
        }
        if (offset + queryLen > maxSeqLen) {
            throw new IllegalArgumentException("offset + queryLen exceeds maxSeqLen");
        }
        if (keyLen > maxSeqLen) {
            throw new IllegalArgumentException("keyLen exceeds maxSeqLen");
        }
        
        try (Tensor queryRows = biasCache.slice(1, offset, offset + queryLen);
             Tensor window = queryRows.slice(2, 0, keyLen)) {
            return FlashAttentionOp.copyOf(window);
        }
    }

    public float[] getSlopes() {
        return slopes.clone();
    }

    public float getSlope(int head) {
        if (head < 0 || head >= numHeads) {
            throw new IndexOutOfBoundsException("Head index out of bounds: " + head);
        }
        return slopes[head];
    }

    private void ensureInitialized() {
        if (biasCache == null) {
            throw new IllegalStateException("ALiBiPE not initialized. Call initialize(backend) first.");
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
        return "ALiBiPE";
    }

    public int numHeads() {
        return numHeads;
    }

    public void close() {
        if (biasCache != null) {
            biasCache.close();
            biasCache = null;
        }
    }

    @Override
    public String toString() {
        return "ALiBiPE{" +
            "dModel=" + dModel +
            ", maxSeqLen=" + maxSeqLen +
            ", numHeads=" + numHeads +
            ", initialized=" + (biasCache != null) +
            '}';
    }
}
