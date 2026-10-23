package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Rotary position embedding. {@link #initialize} uploads cos and sin caches {@code [maxSeqLen, rotaryDim / 2]} with
 * {@code angle = pos / base^(2i / rotaryDim)}; {@link #apply} rotates each pair {@code (x[i], x[i + rotaryDim / 2])}
 * of the first {@code rotaryDim} features of position {@code offset + s} on the input's backend:
 * {@code x[i] * cos - x[i + h] * sin} and {@code x[i] * sin + x[i + h] * cos}, each product and sum rounded once as
 * in Java float arithmetic. Features from {@code rotaryDim} on pass through unchanged.
 *
 * <p>The caches are constants. {@link #apply} is a composition of recording Tensor views and {@link TensorOps} calls,
 * so in grad mode its result is recorded and the backward pass gives the input the inverse rotation of the upstream
 * gradient (and passes the unrotated features' gradient through).</p>
 */
public final class RotaryPE implements PositionalEncoding {
    private final int dModel;
    private final int maxSeqLen;
    private final int rotaryDim;
    private final float base;
    private Tensor cosCache;
    private Tensor sinCache;

    public RotaryPE(int dModel, int maxSeqLen) {
        this(dModel, maxSeqLen, dModel, 10000.0f);
    }

    public RotaryPE(int dModel, int maxSeqLen, int rotaryDim, float base) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (rotaryDim <= 0 || rotaryDim > dModel) {
            throw new IllegalArgumentException("rotaryDim must be in (0, dModel]: " + rotaryDim);
        }
        if (rotaryDim % 2 != 0) {
            throw new IllegalArgumentException("rotaryDim must be even: " + rotaryDim);
        }
        if (base <= 0) {
            throw new IllegalArgumentException("base must be positive: " + base);
        }
        
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.rotaryDim = rotaryDim;
        this.base = base;
    }

    public void initialize(com.eonmux.neuralfree.acceleration.backend.ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        
        int halfRotaryDim = rotaryDim / 2;
        float[] cosData = new float[maxSeqLen * halfRotaryDim];
        float[] sinData = new float[maxSeqLen * halfRotaryDim];
        
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfRotaryDim; i++) {
                float freq = (float) (1.0 / Math.pow(base, (2.0 * i) / rotaryDim));
                float angle = pos * freq;
                int idx = pos * halfRotaryDim + i;
                cosData[idx] = (float) Math.cos(angle);
                sinData[idx] = (float) Math.sin(angle);
            }
        }
        
        close();
        Shape cacheShape = Shape.of(maxSeqLen, halfRotaryDim);
        try (Tensor cosFlat = Tensor.fromArray(backend, cosData);
             Tensor sinFlat = Tensor.fromArray(backend, sinData)) {
            cosCache = cosFlat.reshape(cacheShape);
            sinCache = sinFlat.reshape(cacheShape);
        } catch (RuntimeException e) {
            close();
            throw e;
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
        
        int axis = rank - 1;
        int halfRotaryDim = rotaryDim / 2;
        List<Tensor> parts = new ArrayList<>(3);
        try (Tensor x1 = input.slice(axis, 0, halfRotaryDim);
             Tensor x2 = input.slice(axis, halfRotaryDim, rotaryDim);
             Tensor cos = cosCache.slice(0, offset, offset + seqLen);
             Tensor sin = sinCache.slice(0, offset, offset + seqLen);
             Tensor x1Cos = TensorOps.mul(x1, cos);
             Tensor x2Sin = TensorOps.mul(x2, sin);
             Tensor x1Sin = TensorOps.mul(x1, sin);
             Tensor x2Cos = TensorOps.mul(x2, cos)) {
            parts.add(TensorOps.sub(x1Cos, x2Sin));
            parts.add(TensorOps.add(x1Sin, x2Cos));
            if (rotaryDim < dModel) {
                parts.add(input.slice(axis, rotaryDim, dModel));
            }
            return TensorOps.cat(parts, axis);
        } finally {
            for (Tensor part : parts) {
                part.close();
            }
        }
    }

    private void ensureInitialized() {
        if (cosCache == null || sinCache == null) {
            throw new IllegalStateException("RotaryPE not initialized. Call initialize(backend) first.");
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
        return "RotaryPE";
    }

    public int rotaryDim() {
        return rotaryDim;
    }

    public float base() {
        return base;
    }

    public Tensor getCosCache() {
        ensureInitialized();
        return cosCache;
    }

    public Tensor getSinCache() {
        ensureInitialized();
        return sinCache;
    }

    public void close() {
        if (cosCache != null) {
            cosCache.close();
            cosCache = null;
        }
        if (sinCache != null) {
            sinCache.close();
            sinCache = null;
        }
    }

    @Override
    public String toString() {
        return "RotaryPE{" +
            "dModel=" + dModel +
            ", maxSeqLen=" + maxSeqLen +
            ", rotaryDim=" + rotaryDim +
            ", base=" + base +
            ", initialized=" + (cosCache != null) +
            '}';
    }
}
